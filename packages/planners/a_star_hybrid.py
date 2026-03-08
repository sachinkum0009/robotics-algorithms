"""
Hybrid A* Planner
-----------------
Plans a kinematically-feasible path for a vehicle in a 2-D occupancy grid.

State space  : continuous (x, y, θ)
Motion model : unicycle — arcs produced by discrete steering-angle inputs
Heuristic    : Euclidean distance to the goal (admissible)
Cycle check  : states are snapped to a coarse (x, y, θ) grid
"""

import heapq
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import count
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )

from planners.planner import BasePlanner

# ---------------------------------------------------------------------------
# Discretisation resolution for the visited-cell check
# ---------------------------------------------------------------------------
XY_RES: float = 1.0  # metres
THETA_RES: float = math.radians(15)  # radians


@dataclass
class HybridState:
    """Continuous vehicle pose with search-graph bookkeeping."""

    x: float
    y: float
    theta: float  # heading in radians
    g: float = 0.0  # cost from start
    parent: Optional["HybridState"] = field(default=None, repr=False)

    # ---- comparison / hashing (for heap and visited set) ----

    def __lt__(self, _) -> bool:
        return False  # cost tie-breaking handled by the counter in the heap

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HybridState):
            return self._key() == other._key()
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._key())

    def _key(self) -> tuple:
        """Snap to a coarse grid for cycle detection."""
        n_theta = round(2 * math.pi / THETA_RES)
        return (
            round(self.x / XY_RES),
            round(self.y / XY_RES),
            round(self.theta / THETA_RES) % n_theta,
        )


class HybridAStarPlanner(BasePlanner):
    """
    Hybrid A* planner.

    Parameters
    ----------
    occupancy_grid:
        2-D boolean array – True = obstacle, False = free.
        Row 0 corresponds to y = 0 (origin at bottom-left).
    step_size:
        Arc length of each motion primitive [m].
    steering_angles:
        Discrete steering inputs expressed as heading-change per step [rad].
        Positive = turn left.
    """

    def __init__(
        self,
        occupancy_grid: np.ndarray,
        step_size: float = 1.0,
        steering_angles: Optional[list[float]] = None,
    ) -> None:
        self.grid = occupancy_grid
        self.step_size = step_size
        self.steering_angles = steering_angles or [
            math.radians(a) for a in (-40, -20, 0, 20, 40)
        ]

    # ------------------------------------------------------------------
    # BasePlanner interface
    # ------------------------------------------------------------------

    def plan(self, start: HybridState, goal: HybridState) -> list[HybridState]:
        """Return a list of HybridStates from *start* to *goal* (or [] if none)."""
        tie = count()
        open_heap: list = []
        heapq.heappush(open_heap, (self._f(start, goal), next(tie), start))

        visited: set[tuple] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            key = current._key()
            if key in visited:
                continue
            visited.add(key)

            if self._reached_goal(current, goal):
                return self._reconstruct(current)

            for successor in self._expand(current):
                if successor._key() not in visited:
                    heapq.heappush(
                        open_heap,
                        (self._f(successor, goal), next(tie), successor),
                    )

        return []  # goal unreachable

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _heuristic(self, state: HybridState, goal: HybridState) -> float:
        """Admissible Euclidean distance heuristic."""
        return math.hypot(goal.x - state.x, goal.y - state.y)

    def _f(self, state: HybridState, goal: HybridState) -> float:
        return state.g + self._heuristic(state, goal)

    def _reached_goal(self, state: HybridState, goal: HybridState) -> bool:
        pos_ok = math.hypot(state.x - goal.x, state.y - goal.y) < XY_RES
        heading_ok = abs(_angle_diff(state.theta, goal.theta)) < THETA_RES * 2
        return pos_ok and heading_ok

    def _expand(self, state: HybridState) -> list[HybridState]:
        """Apply each steering input and return collision-free successors."""
        successors = []
        for steer in self.steering_angles:
            nx, ny, ntheta = _kinematic_step(
                state.x, state.y, state.theta, steer, self.step_size
            )
            if self._is_free(nx, ny):
                successors.append(
                    HybridState(
                        x=nx,
                        y=ny,
                        theta=ntheta % (2 * math.pi),
                        g=state.g + self.step_size,
                        parent=state,
                    )
                )
        return successors

    def _is_free(self, x: float, y: float) -> bool:
        """True when (x, y) is inside the grid and not an obstacle."""
        xi, yi = int(round(x)), int(round(y))
        rows, cols = self.grid.shape
        if xi < 0 or xi >= cols or yi < 0 or yi >= rows:
            return False
        return not self.grid[yi, xi]

    @staticmethod
    def _reconstruct(state: HybridState) -> list[HybridState]:
        path: list[HybridState] = []
        cur: Optional[HybridState] = state
        while cur is not None:
            path.append(cur)
            cur = cur.parent
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Pure functions (kinematic model + angle helper)
# ---------------------------------------------------------------------------


def _kinematic_step(
    x: float, y: float, theta: float, steer: float, step: float
) -> tuple[float, float, float]:
    """
    Unicycle model: integrate one arc of length *step*.

    *steer* is the total heading change [rad] over the arc
    (positive = counter-clockwise).
    """
    if abs(steer) < 1e-6:  # straight line
        return (
            x + step * math.cos(theta),
            y + step * math.sin(theta),
            theta,
        )
    # Turning radius derived from arc length and heading change
    R = step / steer
    # Instantaneous centre of curvature
    cx = x - R * math.sin(theta)
    cy = y + R * math.cos(theta)
    # New pose after rotating by *steer* around the ICC
    nx = cx + R * math.sin(theta + steer)
    ny = cy - R * math.cos(theta + steer)
    ntheta = theta + steer
    return nx, ny, ntheta


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference in (-π, π]."""
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return d


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_hybrid_astar(
    grid: np.ndarray,
    path: list[HybridState],
    start: HybridState,
    goal: HybridState,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    # Occupancy grid
    display = np.where(grid[:, :, None], 0.2, 0.95) * np.ones((1, 1, 3))
    ax.imshow(
        display,
        origin="lower",
        extent=[-0.5, grid.shape[1] - 0.5, -0.5, grid.shape[0] - 0.5],
    )

    # Path line
    if path:
        xs = [s.x for s in path]
        ys = [s.y for s in path]
        ax.plot(xs, ys, color="tomato", linewidth=2.5, zorder=3, label="path")

        # Heading arrows sampled along the path
        sample_step = max(1, len(path) // 15)
        for s in path[::sample_step]:
            ax.annotate(
                "",
                xy=(
                    s.x + 0.7 * math.cos(s.theta),
                    s.y + 0.7 * math.sin(s.theta),
                ),
                xytext=(s.x, s.y),
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5),
                zorder=4,
            )

    # Start / goal markers
    ax.scatter(
        start.x, start.y, s=200, color="limegreen", zorder=5, label="start"
    )
    ax.scatter(
        goal.x, goal.y, s=200, color="gold", marker="*", zorder=5, label="goal"
    )

    title_path = (
        f"({path[0].x:.1f},{path[0].y:.1f}) → ({path[-1].x:.1f},{path[-1].y:.1f}), "
        f"{len(path)} steps"
        if path
        else "No path found"
    )
    ax.set_title(f"Hybrid A*\n{title_path}", fontsize=13)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    # 20 × 20 free grid with a vertical wall obstacle
    grid = np.zeros((20, 20), dtype=bool)
    grid[4:16, 10] = True  # vertical wall blocking the middle

    start = HybridState(x=2.0, y=10.0, theta=0.0)
    goal = HybridState(x=17.0, y=10.0, theta=0.0)

    planner = HybridAStarPlanner(grid, step_size=1.0)
    path = planner.plan(start, goal)

    if path:
        print(f"Path found: {len(path)} states")
        for s in path:
            print(
                f"  x={s.x:6.2f}  y={s.y:6.2f}  θ={math.degrees(s.theta):7.2f}°"
            )
    else:
        print("No path found.")

    plot_hybrid_astar(grid, path, start, goal)


if __name__ == "__main__":
    main()
