from typing import Optional, List, Tuple, Union
from gym_multigrid.multigrid import World, Agent, Grid, Ball, MultiGridEnv, Wall
import numpy as np

class HarvestGameEnv(MultiGridEnv):
    """
    CPR problem with replenishment
    """
    def __init__(
            self,
            size: int = 10,
            width: Optional[int] = None,
            height: Optional[int] = None,
            init_resource: List[int] = list(),
            agents_idx: List[int] = list(),
            resource_idx: List[int] = list(),
            resource_reward: List[int] = list(),
            zero_sum: bool = False,
            agent_view_size: int = 7,
    ):
        self.init_resource = init_resource
        self.resource_idx = resource_idx
        self.resource_reward = resource_reward
        self.zero_sum = zero_sum

        self.world = World

        agents = [Agent(self.world, i, view_size=agent_view_size) for i in agents_idx]

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=1e4,
            # set this to true for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=agent_view_size
        )

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)

        # generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height - 1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width - 1, 0)

        for number, idx, reward in zip(self.init_resource, self.resource_idx, self.resource_reward):
            for i in range(number):
                self.place_obj(Ball(self.world, idx, reward))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)

    def _reward(self, i: int, rewards: List[int], reward: int = 1):
        """Compute the reward to be given upon success"""
        for j, a in enumerate(self.agents):
            if a.index == i or a.index == 0:
                rewards[j] += reward
            if self.zero_sum:
                if a.index != i or a.index == 0:
                    rewards[j] -= reward

    def _handle_pickup(self, i: int, rewards: List[int], fwd_pos: Tuple[int, int], fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if fwd_cell.index in [0, self.agents[i].index]:
                    fwd_cell.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    self._reward(i, rewards, fwd_cell.reward)

    def _handle_drop(self, i: int, rewards: List[int], fwd_pos: Tuple[int, int], fwd_cell):
        pass

    @property
    def radius(self) -> int:
        """Will fail if called before _gen_grid"""
        return (self.grid.width + self.grid.height) // 6

    @property
    def middle(self) -> Tuple[int, int]:
        """Will fail if called before _gen_grid"""
        return (self.grid.width // 2, self.grid.height // 2)

    @staticmethod
    def neighbs(x: int, y: int) -> List[Tuple[int, int]]:
       return [
            (x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
            (x, y - 1), (x, y + 1),
            (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)
        ]

    def _rand_neighbor(self, x: int, y: int, max_tries: float = np.inf) -> Optional[Tuple[int, int]]:
        neighbs = self.neighbs(x, y)
        # neighbs = self.eps_neighb(self.radius, (x,y))

        num_tries = 0
        while True:
            if max_tries < num_tries:
                return None
            neighb = self._rand_elem(neighbs)
            if neighb[0] >= self.grid.width or neighb[1] >= self.grid.height:
                num_tries += 1
                continue
            else:
                return neighb

    @staticmethod
    def eps_neighb(epsilon: int, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        pass

    def _rand_rim(
            self,
            top: Optional[Tuple[int, int]] = None,
            size: Optional[Tuple[int, int]] = None,
            reject_fn: Optional[Callable[[...], bool]] = None,
            max_tries: float = np.inf
    ) -> Tuple[int, int]:
        """Returns a random position near the rim of the world"""

        def eps_neighb_member(epsilon: int, pos: Tuple[int, int], x0: Tuple[int, int]) -> bool:
            """
            is x0 in the discrete epsilon-neighborhood about pos
            actually this makes a box rather than a ball :(
            """
            x, y = pos
            x0, y0 = x0
            return any([
                x0 <= x - epsilon,
                x0 >= x + epsilon,
                y0 <= y - epsilon,
                y0 >= y + epsilon
            ])

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0
        while True:
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object in a radius about the center
            if eps_neighb_member(self.radius, self.middle, pos):
                continue
            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        return pos

    def place_obj(
            self,
            obj: Optional[Union[Agent, Wall, Ball]],
            top: Optional[Tuple[int, int]] = None,
            size: Optional[Tuple[int, int]] = None,
            reject_fn: Optional[Callable[[...], bool]] = None,
            max_tries: float = np.inf
    ) -> Tuple[int, int]:
        """Place an object at an empty position in the grid"""
        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        if isinstance(obj, Agent):
            num_tries = 0
            while True:
                if num_tries > max_tries:
                    raise RecursionError("rejection sampling failed in place_obj")
                num_tries += 1

                pos = self._rand_rim(top, size)
                # Don't place the agent on top of another agent
                if self.grid.get(*pos) != None:
                    continue
                # Check if there is a filtering criterion
                if reject_fn and reject_fn(self, pos):
                    continue

                break
            self.grid.set(*pos, obj)
            if obj is not None:
                obj.init_pos = pos
                obj.cur_pos = pos
            return pos

        assert isinstance(obj, Ball)

        pos = np.array(self.middle)

        while self.grid.get(*pos) != None:
            pos = self._rand_neighbor(*pos)

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def step(self, actions):
        p = 1.006e-3
        reward = 1
        idx = len(self.resource_idx) + 1
        for i, ob in enumerate(self.grid.grid):
            if ob is None or isinstance(i, (Agent, Wall)):
                continue
            x, y = divmod(i, self.width) # i'm not 100% sure this is correct
            resource_neighbs = list()
            for neighb in self.neighbs(x,y):
                if neighb is None or isinstance(neighb, (Agent, Wall)):
                    continue
                resource_neighbs.append(neighb)
            if self._rand_float(0, 1) > 1 - p * len(resource_neighbs):
                self.place_obj(Ball(self.world))
                idx += 1

        return MultiGridEnv.step(self, actions)

class Harvest4HEnv10x10N2(HarvestGameEnv):
    size = 32
    def __init__(self):
        super().__init__(
            size=self.size,
            init_resource=[self.size // 4],
            agents_idx=[1,2,3],
            resource_idx=[0],
            resource_reward=[1],
            zero_sum=False
        )
