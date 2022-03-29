import operator
from collections import namedtuple
from copy import deepcopy
from typing import Set, Tuple, Sequence, FrozenSet, Iterable, List, Optional

import clingo

from action_language_interpreter.util import symbol_to_str

Event = str
Fluent = str


class FluentLiteral(namedtuple('FluentLiteral', ['name', 'sign'], defaults=[True])):

    def __neg__(self):
        return FluentLiteral(name=self.name, sign=not self.sign)

    def __invert__(self):
        return FluentLiteral(name=self.name, sign=not self.sign)

    def __repr__(self):
        return f"{'' if self.sign else '¬'}{self.name}"


State = Set[FluentLiteral]
ScenarioPath = Tuple[State, List[Tuple[Event, State]]]


def match(fl1: FluentLiteral, fl2: FluentLiteral):
    return fl1 == fl2


def contradict(fl1: FluentLiteral, fl2: FluentLiteral):
    f1, s1 = fl1
    f2, s2 = fl2
    return f1 == f2 and s1 != s2


def state_match(state: State, conditions: Iterable[FluentLiteral]):
    return all(
        not any(contradict(fluent_literal, condition) for condition in conditions) for fluent_literal in state) and all(
        any(match(condition, fluent_literal) for fluent_literal in state) for condition in conditions)


def change_state(state: State, *fluent_literals: FluentLiteral) -> State:
    next_state = deepcopy(state)
    for fluent_literal in fluent_literals:
        fluent, sign = fluent_literal
        for i, state_fluent_literal in enumerate(state):
            state_fluent, state_sign = state_fluent_literal
            if state_fluent == fluent:
                if state_sign != sign:
                    next_state.discard(state_fluent_literal)
                break
        next_state.add(fluent_literal)
    return next_state


class ALStatement:

    def fires(self, state: State, event: Optional[Event] = None) -> bool:
        return False


class DynamicLaw(ALStatement):

    def __init__(self, event: Event, fluent_literal: FluentLiteral, *conditions: FluentLiteral):
        self.event: Event = event
        self.fluent_literal: FluentLiteral = fluent_literal
        self.conditions: FrozenSet[FluentLiteral] = frozenset(conditions)

    def __hash__(self):
        return hash(("Dynamic Law", self.event, self.fluent_literal, self.conditions))

    def fires(self, state: Set[FluentLiteral], event: Optional[Event] = None) -> bool:
        return (event is None or self.event == event) and state_match(state, self.conditions)

    def __repr__(self):
        return f"{self.event} causes {self.fluent_literal} if {','.join(map(repr, self.conditions))}"


class StateConstraint(ALStatement):
    def __init__(self, fluent_literal: FluentLiteral, *conditions: FluentLiteral):
        self.fluent_literal: FluentLiteral = fluent_literal
        self.conditions: FrozenSet[FluentLiteral] = frozenset(conditions)

    def __hash__(self):
        return hash(("State Constraint", self.fluent_literal, self.conditions))

    def fires(self, state: Set[FluentLiteral], event: Optional[Event] = None) -> bool:
        return state_match(state, self.conditions)

    def __repr__(self):
        return f"{self.fluent_literal} if {','.join(map(repr, self.conditions))}"


class ExecutabilityCondition(ALStatement):
    def __init__(self, event: Event, *conditions: FluentLiteral):
        self.event: Event = event
        self.conditions: FrozenSet[FluentLiteral] = frozenset(conditions)

    def __hash__(self):
        return hash(("Executability Condition", self.event, self.conditions))

    def fires(self, state: Set[FluentLiteral], event: Optional[Event] = None) -> bool:
        return (event is None or event == self.event) and state_match(state, self.conditions)

    def __repr__(self):
        return f"{self.event} impossible_if {','.join(map(repr, self.conditions))}"


class ActionDescription:
    def __init__(self):
        self.dynamic_laws: Set[DynamicLaw] = set()
        self.state_constraints: Set[StateConstraint] = set()
        self.executability_conditions: Set[ExecutabilityCondition] = set()

    def __repr__(self):
        return repr((self.dynamic_laws, self.state_constraints, self.executability_conditions))

    def __str__(self):
        return str((self.dynamic_laws, self.state_constraints, self.executability_conditions))

    def add_stmt_causes_if(self, event, fluent_literal, *conditions):
        self.dynamic_laws.add(DynamicLaw(event, fluent_literal, *conditions))

    def add_stmt_if(self, fluent_literal, *conditions):
        self.state_constraints.add(StateConstraint(fluent_literal, *conditions))

    def add_stmt_impossible_if(self, event, *conditions):
        self.executability_conditions.add(ExecutabilityCondition(event, *conditions))

    def executable_events(self, state: Set[FluentLiteral], events: Sequence[Event]) -> Tuple[
        Set[Event], Sequence[ExecutabilityCondition]]:
        possible_events = set()
        relevant_executability_conditions = set()
        for event in events:
            possible = True
            for executablility_condition in self.executability_conditions:
                if executablility_condition.fires(state, event):
                    relevant_executability_conditions.add(executablility_condition)
                    possible = False
            if possible:
                possible_events.add(event)

        return possible_events, tuple(relevant_executability_conditions)

    def execute(self, state: State, event: Event) -> Tuple[State, Sequence[ALStatement]]:
        next_state_pass1, relevant_statements_pass1 = self._execute_dynamic_laws(state, event)
        next_state, relevant_statements = self._execute_state_constraints(next_state_pass1, relevant_statements_pass1)
        return next_state, relevant_statements

    def comply(self, state: State) -> Tuple[State, Sequence[ALStatement]]:
        return self._execute_state_constraints(state, [])

    def _execute_dynamic_laws(self, state: State, event: Event) -> Tuple[State, List[ALStatement]]:
        relevant_dynamic_laws: List[DynamicLaw] = [dynamic_law for dynamic_law in self.dynamic_laws if
                                                   dynamic_law.fires(state, event)]
        changes = {dynamic_law.fluent_literal for dynamic_law in relevant_dynamic_laws}
        next_state_pass1 = change_state(state, *changes)
        return next_state_pass1, relevant_dynamic_laws

    def _execute_state_constraints(self, state, relevant_statements: List[ALStatement]) -> Tuple[
        State, List[ALStatement]]:
        change = True
        next_state = state
        while change:
            change = False
            for state_constraint in self.state_constraints:
                if state_constraint not in relevant_statements and state_constraint.fires(next_state):
                    change = True
                    next_state = change_state(next_state, state_constraint.fluent_literal)
                    relevant_statements.append(state_constraint)
        return next_state, relevant_statements

    def direct_effects(self, event: Event, state: Optional[State] = None) -> Set[FluentLiteral]:
        return {dynamic_law.fluent_literal for dynamic_law in self.dynamic_laws if
                dynamic_law.event == event and (state is None or dynamic_law.fires(state, event))}

    def preconditions(self, event: Event) -> Set[FluentLiteral]:
        preconditions = set()
        for dynamic_law in self.dynamic_laws:
            if dynamic_law.event == event:
                preconditions.update(dynamic_law.conditions)
        for executability_condition in self.executability_conditions:
            if executability_condition.event == event:
                preconditions.update(map(operator.neg, executability_condition.conditions))
        return preconditions


def from_clingo_symbols(symbols:Sequence[clingo.Symbol]) -> ActionDescription:
    ad = ActionDescription()
    for symbol in symbols: # type: clingo.Symbol
        if symbol.type == clingo.SymbolType.Function:
            if symbol.name == "causes":
                event = symbol.arguments[0]
                effect = symbol.arguments[1]
                conditions = symbol.arguments[2:]
                ad.add_stmt_causes_if(symbol_to_str(event), FluentLiteral(symbol_to_str(effect)), *map(FluentLiteral, map(symbol_to_str,conditions)))
            elif symbol.name == "impossible_if":
                event = symbol.arguments[0]
                conditions = symbol.arguments[1:]
                ad.add_stmt_impossible_if(event, *conditions)
            elif symbol.name == "if":
                effect = symbol.arguments[0]
                conditions = symbol.arguments[1:]
                ad.add_stmt_if(effect, *conditions)
    return ad




def get_transition_events(scenario_path: ScenarioPath, outcome: State) -> Set[Event]:
    initial_state, event_chain = scenario_path
    transition_events = set()
    previous_state = initial_state
    for (event, current_state) in event_chain:
        if not (outcome <= previous_state) and (outcome <= current_state):
            transition_events.add(event)
        previous_state = current_state
    return transition_events


def get_effects(action_description: ActionDescription, scenario_path: ScenarioPath, outcome: Optional[State] = None,
                transition_event: Optional[Event] = None) -> Tuple[
    Set[FluentLiteral], Set[FluentLiteral]]:
    initial_state: State
    event_chain: List[Tuple[Event, State]]
    initial_state, event_chain = scenario_path
    if outcome is None:
        _, outcome = event_chain[-1]
    previous_state: State = initial_state
    for (event, current_state) in event_chain:
        if transition_event is None or transition_event == event:
            if not (outcome <= previous_state) and (outcome <= current_state):
                inertial_literals = previous_state & current_state
                direct_effects = action_description.direct_effects(event, previous_state)
                indirect_effects = outcome & (current_state - (direct_effects | inertial_literals))
                return direct_effects, indirect_effects
        previous_state = current_state


get_first_causal_explanation = get_effects


def get_ensuring_event(scenario_path: ScenarioPath, fluent_literal: FluentLiteral, state: Optional[State] = None) -> \
        Optional[Event]:
    ensuring_event_candidates = get_transition_events(scenario_path, {fluent_literal})
    initial_state, event_chain = scenario_path
    for (event, current_state) in reversed(event_chain):
        if event in ensuring_event_candidates and (state is None or current_state == state):
            return event
    return None


def get_second_causal_explanation(scenario_path: ScenarioPath, outcome: Optional[State] = None,
                                  action_description: Optional[ActionDescription] = None,
                                  effects: Optional[Set[FluentLiteral]] = None) -> Tuple[
    Set[Tuple[Event, FluentLiteral]], Set[FluentLiteral]]:
    if effects is None:
        if action_description is None:
            raise TypeError("If effects are not explicitly given an action description has to be given")
        direct_effects, indirect_effects = get_effects(action_description=action_description,
                                                       scenario_path=scenario_path,
                                                       outcome=outcome)
        effects = direct_effects | indirect_effects
    if outcome is None:
        initial_state, event_chain = scenario_path
        transition_event, outcome = event_chain[-1]
    remaining_outcome_literals = outcome - effects
    supporting_literals = set()
    initial_literals = set()
    for remaining_outcome_literal in remaining_outcome_literals:
        ensuring_event = get_ensuring_event(scenario_path, remaining_outcome_literal)
        if ensuring_event is None:
            initial_literals.add(remaining_outcome_literal)
        else:
            supporting_literals.add((ensuring_event, remaining_outcome_literal))

    return supporting_literals, initial_literals


def get_third_causal_explanation(action_description: ActionDescription, scenario_path: ScenarioPath,
                                 outcome: Optional[State] = None, transition_event: Optional[Event] = None) -> Tuple[
    Set[Tuple[Event, FluentLiteral]], Set[FluentLiteral]]:
    initial_state, event_chain = scenario_path
    if transition_event is None and outcome is None:
        transition_event, outcome = event_chain[-1]
    elif transition_event is None or outcome is None:
        for (event, state) in reversed(event_chain):
            if outcome is None:
                if event == transition_event:
                    outcome = state
                    break
            elif transition_event is None:
                if state == outcome:
                    transition_event = event
                    break

    preconditions = action_description.preconditions(transition_event)
    supporting_events = set()
    uncaused_literals = set()
    for precondition in preconditions:
        ensuring_event = get_ensuring_event(scenario_path, precondition)
        if ensuring_event is None:
            uncaused_literals.add(precondition)
        else:
            supporting_events.add((ensuring_event, precondition))

    return supporting_events, uncaused_literals
