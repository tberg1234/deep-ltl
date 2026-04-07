import random
from typing import Callable

from ltl.automata import LDBASequence
from ltl.logic import Assignment
from envs.gym_repoman.envs.collect_env import CollectEnv

_env = CollectEnv()
_assignments = _env.get_possible_assignments()
_assignments.remove(Assignment.zero_propositions(_env.get_propositions()))


def all_reach_tasks_repoman(depth: int) -> Callable[[list[str]], list[LDBASequence]]:
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        reachs = [
            (frozenset([a.to_frozen()]), frozenset())
            for a in _assignments
        ]

        def rec(d):
            if d == 1:
                return [[r] for r in reachs]
            result = []
            for task in rec(d - 1):
                next_reach = task[0][0]
                for r, a in reachs:
                    if r == next_reach:
                        continue
                    result.append([(r, a)] + task)
            return result

        return [LDBASequence(task) for task in rec(depth)]

    return wrapper


def all_reach_avoid_tasks_repoman(depth: int) -> Callable[[list[str]], list[LDBASequence]]:
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        reach_avoids = [
            (
                frozenset([a1.to_frozen()]),
                frozenset([a2.to_frozen()]),
            )
            for a1 in _assignments
            for a2 in _assignments
            if a1 != a2
        ]

        def rec(d):
            if d == 1:
                return [[ra] for ra in reach_avoids]
            result = []
            for task in rec(d - 1):
                next_reach, next_avoid = task[0]
                for p, q in reach_avoids:
                    if p == next_reach or p == next_avoid:
                        continue
                    result.append([(p, q)] + task)
            return result

        return [LDBASequence(task) for task in rec(depth)]

    return wrapper


def sample_reach_avoid_repoman(
        depth: int | tuple[int, int],
        num_reach: int | tuple[int, int],
        num_avoid: int | tuple[int, int],
) -> Callable[[list[str]], LDBASequence]:
    def wrapper(propositions: list[str]) -> LDBASequence:
        def sample_one(last_assignment):
            available = [a for a in _assignments if a != last_assignment]
            nr = random.randint(*num_reach) if isinstance(num_reach, tuple) else num_reach
            na = random.randint(*num_avoid) if isinstance(num_avoid, tuple) else num_avoid
            reach_assignments = random.sample(available, min(nr, len(available)))
            avoid_available = [a for a in available if a not in reach_assignments]
            na = min(na, len(avoid_available))
            avoid_assignments = random.sample(avoid_available, na)
            reach = frozenset([a.to_frozen() for a in reach_assignments])
            avoid = frozenset([a.to_frozen() for a in avoid_assignments])
            return reach, avoid, reach_assignments[0]

        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        last_assignment = None
        seq = []
        for _ in range(d):
            reach, avoid, last_assignment = sample_one(last_assignment)
            seq.append((reach, avoid))
        return LDBASequence(seq)

    return wrapper
