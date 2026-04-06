import random
from typing import Callable

from ltl.automata import LDBASequence
from ltl.logic import Assignment

# Matches CollectEnv._AVAILABLE_COLLECTIBLES
_COLLECTIBLES = [
    ('square', 'purple'),
    ('circle', 'purple'),
    ('square', 'beige'),
    ('circle', 'beige'),
    ('square', 'blue'),
    ('circle', 'blue'),
]


def _pair_frozen(shape: str, colour: str, propositions: list[str]):
    true_props = {shape, colour}
    return Assignment({p: (p in true_props) for p in propositions}).to_frozen()


def all_reach_tasks_repoman(depth: int) -> Callable[[list[str]], list[LDBASequence]]:
    def wrapper(propositions: list[str]) -> list[LDBASequence]:
        reachs = [
            (frozenset([_pair_frozen(s, c, propositions)]), frozenset())
            for s, c in _COLLECTIBLES
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
                frozenset([_pair_frozen(s1, c1, propositions)]),
                frozenset([_pair_frozen(s2, c2, propositions)]),
            )
            for s1, c1 in _COLLECTIBLES
            for s2, c2 in _COLLECTIBLES
            if (s1, c1) != (s2, c2)
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
        def sample_one(last_pair):
            available = [pair for pair in _COLLECTIBLES if pair != last_pair]
            nr = random.randint(*num_reach) if isinstance(num_reach, tuple) else num_reach
            na = random.randint(*num_avoid) if isinstance(num_avoid, tuple) else num_avoid
            reach_pairs = random.sample(available, min(nr, len(available)))
            avoid_available = [pair for pair in available if pair not in reach_pairs]
            na = min(na, len(avoid_available))
            avoid_pairs = random.sample(avoid_available, na)
            reach = frozenset([_pair_frozen(s, c, propositions) for s, c in reach_pairs])
            avoid = frozenset([_pair_frozen(s, c, propositions) for s, c in avoid_pairs])
            return reach, avoid, reach_pairs[0]

        d = random.randint(*depth) if isinstance(depth, tuple) else depth
        last_pair = None
        seq = []
        for _ in range(d):
            reach, avoid, last_pair = sample_one(last_pair)
            seq.append((reach, avoid))
        return LDBASequence(seq)

    return wrapper
