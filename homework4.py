import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from fontTools.t1Lib import std_subrs

import environment

class CNP(torch.nn.Module):
    def __init__(self, x_dim, y_dim, h_dim, hidden_size=128, num_hidden_layers=2, min_std=0.1):
        super(CNP, self).__init__()
        self.d_x = x_dim
        self.d_y = y_dim
        self.d_h = h_dim # dimension of the context parameter (height of the object for our experiment)

        self.encoder = []
        self.encoder.append(torch.nn.Linear(self.d_x + self.d_y + self.d_h, hidden_size))
        self.encoder.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
        self.encoder = torch.nn.Sequential(*self.encoder)

        self.query = []
        self.query.append(torch.nn.Linear(hidden_size + self.d_x, hidden_size))
        self.query.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.query.append(torch.nn.Linear(hidden_size, hidden_size))
            self.query.append(torch.nn.ReLU())
        self.query.append(torch.nn.Linear(hidden_size, 2 * self.d_y))
        self.query = torch.nn.Sequential(*self.query)

        self.min_std = min_std

    def nll_loss(self, observation, target, target_truth, observation_mask=None, target_mask=None):
        '''
        The original negative log-likelihood loss for training CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor that contains context
            points.
            d_x: the number of query dimensions
            d_y: the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor that contains query dimensions
            of target (query) points.
            d_x: the number of query dimensions.
            note: n_context and n_target does not need to be the same size.
        target_truth : torch.Tensor
            (n_batch, n_target, d_y) sized tensor that contains target
            dimensions (i.e., prediction dimensions) of target points.
            d_y: the number of target dimensions
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation. Used for batch input.
        target_mask : torch.Tensor
            (n_batch, n_target) sized tensor indicating which entries should be
            used for loss calculation. Used for batch input.
        Returns
        -------
        loss : torch.Tensor (float)
            The NLL loss.
        '''
        mean, std = self.forward(observation, target, observation_mask)
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            # sum over the sequence (i.e. targets in the sequence)
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            # compute the number of entries for each batch entry
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            # first normalize, then take an average over the batch and dimensions
            loss = (nll_masked / nll_norm).mean()
        else:
            loss = nll.mean()
        return loss

    def forward(self, observation, target, observation_mask=None):
        '''
        Forward pass of CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor where d_x is the number
            of the query dimensions, d_y is the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor where d_x is the number of
            query dimensions. n_context and n_target does not need to be the
            same size.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation.
        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean
            prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard
            deviation prediction.
        '''
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        h_cat = self.concatenate(r, target)
        query_out = self.decode(h_cat)
        mean = query_out[..., :self.d_y]
        logstd = query_out[..., self.d_y:]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std

    def encode(self, observation):
        h = self.encoder(observation)
        return h

    def decode(self, h):
        o = self.query(h)
        return o

    def aggregate(self, h, observation_mask):
        # this operation is equivalent to taking mean but for
        # batched input with arbitrary lengths at each entry
        # the output should have (batch_size, dim) shape

        if observation_mask is not None:
            h = (h * observation_mask.unsqueeze(2)).sum(dim=1)  # mask unrelated entries and sum
            normalizer = observation_mask.sum(dim=1).unsqueeze(1)  # compute the number of entries for each batch entry
            r = h / normalizer  # normalize
        else:
            # if observation mask is none, we assume that all entries
            # in the batch has the same length
            r = h.mean(dim=1)
        return r

    def concatenate(self, r, target):
        num_target_points = target.shape[1]
        r = r.unsqueeze(1).repeat(1, num_target_points, 1)  # repeating the same r_avg for each target
        h_cat = torch.cat([r, target], dim=-1)
        return h_cat

    def to(self, device):
        super().to(device)
        self.encoder = self.encoder.to(device)
        self.query = self.query.to(device)



class Hw5Env(environment.BaseEnv):
    def __init__(self, render_mode="gui") -> None:
        self._render_mode = render_mode
        self.viewer = None
        self._init_position = [0.0, -np.pi/2, np.pi/2, -2.07, 0, 0, 0]
        self._joint_names = [
            "ur5e/shoulder_pan_joint",
            "ur5e/shoulder_lift_joint",
            "ur5e/elbow_joint",
            "ur5e/wrist_1_joint",
            "ur5e/wrist_2_joint",
            "ur5e/wrist_3_joint",
            "ur5e/robotiq_2f85/right_driver_joint"
        ]
        self.reset()
        self._joint_qpos_idxs = [self.model.joint(x).qposadr for x in self._joint_names]
        self._ee_site = "ur5e/robotiq_2f85/gripper_site"

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [0.5, 0.0, 1.5]
        height = np.random.uniform(0.03, 0.1)
        self.obj_height = height
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, height], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="frontface")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=0).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[1:]
        obj_pos = self.data.body("obj1").xpos[1:]
        return np.concatenate([ee_pos, obj_pos, [self.obj_height]])


def bezier(p, steps=100):
    t = np.linspace(0, 1, steps).reshape(-1, 1)
    curve = np.power(1-t, 3)*p[0] + 3*np.power(1-t, 2)*t*p[1] + 3*(1-t)*np.power(t, 2)*p[2] + np.power(t, 3)*p[3]
    return curve

def get_training_data(traj_batch):
    traj_batch_size = len(traj_batch)
    traj_len = traj_batch.shape[1]  # all trajectories have the same length (100)

    obs_per_traj = np.random.randint(1, max_obs_per_traj + 1)
    obs_indices = torch.randint(0, traj_len, size=(traj_batch_size, obs_per_traj), device=device)
    batch_indices = torch.arange(traj_batch_size, device=device).unsqueeze(1)
    obs_values = traj_batch[batch_indices, obs_indices]
    obs_t = obs_indices.float() / traj_len
    obs = torch.cat([obs_t.unsqueeze(-1), obs_values], dim=-1)  # append time to observations
    obs = obs.to(device)

    target_indices = torch.randint(0, traj_len, size=(traj_batch_size, 1), device=device)
    target_t = target_indices.float() / traj_len
    target_t = target_t.unsqueeze(-1)  # shape is (batch_size, n_target=1, d_x=1)
    target_values = traj_batch[batch_indices, target_indices]
    target_values = target_values[:, :, :-1]  # remove the last dimension (context variable, only necessary as input)
    target_values = target_values.to(device)

    return obs, target_t, target_values



if __name__ == "__main__":
    env = Hw5Env(render_mode="offscreen")

    # collecting trajectories

    use_existing_trajectories = True
    # use_existing_trajectories = False

    if use_existing_trajectories:
        trajectory_save_path = "HW4/trajectories.npy"
        states_arr = np.load(trajectory_save_path)
        print(f"Loaded {len(states_arr)} trajectories.")
    else:
        trajectory_save_path = "HW4/trajectories.npy"
        trajectory_save_file = open(trajectory_save_path, "w")
        states_arr = []
        for i in range(200):
            env.reset()
            p_1 = np.array([0.5, 0.3, 1.04])
            p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
            p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
            p_4 = np.array([0.5, -0.3, 1.04])
            points = np.stack([p_1, p_2, p_3, p_4], axis=0)
            curve = bezier(points)

            env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
            states = []
            for p in curve:
                env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
                states.append(env.high_level_state())
            states = np.stack(states)
            states_arr.append(states)
            print(f"Collected {i+1} trajectories.", end="\r")
        # save all trajectories to trajectory_save_file
        states_arr = np.stack(states_arr)
        np.save(trajectory_save_path, states_arr)


    training_trajectories = states_arr[:100]
    print(f"Training on {len(training_trajectories)} trajectories.")
    validation_trajectories = states_arr[100:]
    validation_n = len(validation_trajectories)
    validation_trajectories = validation_trajectories[validation_n // 5:]
    test_trajectories = validation_trajectories[:validation_n // 5]
    print(f"Validating on {len(validation_trajectories)} trajectories.")

    best_model_path = "HW4/best_model.pth"

    model = CNP(x_dim=1, y_dim=4, h_dim=1) # x_dim: t. y_dim: yz of ee, yz of object

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # model and training hyperparameters
    traj_batch_size = 64
    num_steps = 10 ** 6
    max_obs_per_traj = 10

    # train CNMP with trajectories to predict y of target point given some observations
    train_new_model = True
    # train_new_model = False
    if train_new_model:
        validation_best = np.inf
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        training_losses = []
        validation_losses = []
        for i in range(num_steps):
            # high_level_state: [ee_y, ee_z, obj_y, obj_z, obj_height]

            traj_batch_indices = np.random.choice(len(states_arr), traj_batch_size)
            traj_batch = states_arr[traj_batch_indices]
            traj_batch = torch.tensor(traj_batch, dtype=torch.float32).to(device)

            obs, target_t, target_values = get_training_data(traj_batch)

            loss = model.nll_loss(obs, target_t, target_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f"Step {i}, Training Loss: {loss.item()}")
                training_losses.append(loss.item())
                # Validation
                with torch.no_grad():
                    val_loss = 0
                    val_traj_idx = np.random.choice(len(validation_trajectories), 1)
                    val_traj = validation_trajectories[val_traj_idx]
                    val_traj = torch.tensor(val_traj, dtype=torch.float32).to(device)
                    obs, target_t, target_values = get_training_data(val_traj)
                    val_loss = model.nll_loss(obs, target_t, target_values)
                    print(f"Validation Loss: {val_loss.item()}")
                    validation_losses.append(val_loss.item())
                    if val_loss < validation_best:
                        validation_best = val_loss
                        torch.save(model.state_dict(), best_model_path)
                        print(f"New validation best. Model saved.")

    # load best model
    model.load_state_dict(torch.load(best_model_path))

    # test model
    print(f"Testing on {len(test_trajectories)} trajectories, 5 tests from each.")
    mean_errors = []
    std_errors = []
    for test_traj in test_trajectories:
        test_traj = torch.tensor(test_traj, dtype=torch.float32)
        for i in range(5):
            obs, target_t, target_values = get_training_data(test_traj)
            with torch.no_grad():
                mean, std = model(obs, target_t)
            mean_errors.append((mean - target_values).abs().mean().item())
            std_errors.append(std.mean().item())

    mean_errors = np.array(mean_errors)
    std_errors = np.array(std_errors)

    print(f"Mean error: {mean_errors.mean()}")
    print(f"Std error: {std_errors.mean()}")

    # plot the errors in a bar plot with mean being the height of the bar, and std being a line indicator on the bar
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(mean_errors)), mean_errors, yerr=std_errors, capsize=5)
    ax.set_xlabel("Test Trajectory")
    ax.set_ylabel("Mean Error")
    ax.set_title("Mean Error with Std Error")
    plt.show()


    # plotting trajectories
    fig, ax = plt.subplots(1, 2)
    for states in states_arr:
        ax[0].plot(states[:, 0], states[:, 1], alpha=0.2, color="b")
        ax[0].set_xlabel("e_y")
        ax[0].set_ylabel("e_z")

        # plot predicted object positions here

        ax[1].plot(states[:, 2], states[:, 3], alpha=0.2, color="r")
        ax[1].set_xlabel("o_y")
        ax[1].set_ylabel("o_z")

        # plot predicted object positions here

    plt.show()
