%conditionFails(C, T) :- time(T), condition_fluent(C, F), not pos_at(F, T).
conditionFails(C, T) :- time(T), condition_fluent(C, F), neg_at(F, T).
%conditionFails(C, T) :- time(T), condition_fluent(C, -F), not neg_at(F, T).
conditionFails(C, T) :- time(T), condition_fluent(C, -F), pos_at(F, T).

obs_at(F, T) :- event_causes_if(E, F, C), occ_at(E, T), not conditionFails(C, T).