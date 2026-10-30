
# Phase 0 — Business Framing and Evaluation Protocol

## Business Decision

Choose which mobile-money transactions should be approved, blocked,
subjected to additional verification, or sent for manual review before
approval.

The objective is to reduce modeled fraud costs while respecting
review capacity and minimizing unnecessary customer friction.

## Scoring Moment

Before transaction approval.

The deployable feature set may only use information available at
decision time.

## Initial Cost Assumptions

All amounts below are hypothetical PaySim amount-unit assumptions,
not verified real-world operating costs.

- Manual review: 100 units per reviewed alert.
- Missed fraud: transaction amount used as a proxy for financial
  exposure, not as confirmed money lost.
- Incorrectly blocking a legitimate transaction: 500 units.
- Additional verification: 25 units per attempt.

These assumptions will be subjected to sensitivity analysis.

## Operational Constraints

The primary scenario allows up to 500 manual-review alerts per day.
Additional scenarios will consider 100 and 1,000 alerts per day.

## Initial Success Criteria

1. Remain within the alert budget.
2. Produce positive modeled net benefit under the base-case assumptions.
3. Require a machine-learning model to identify at least 5% more
   attempted fraud transaction value than the best feasible rule
   under the same alert budget.
4. Require the model to produce higher modeled net benefit than
   the best feasible rule before preferring it over that rule.
5. Use only decision-time features for the deployable system.
6. Select the policy on validation data and score the locked test
   block only once.

The 5% hurdle, capacity limits, and cost values are project
assumptions, not universal industry standards.

## Analytical Questions

1. How much fraud occurs?
2. Where does fraud occur?
3. When does fraud occur?
4. What behavioral patterns are associated with fraud?
5. How reliable are the data and conclusions?
6. What value does a model add over simpler rules?
7. What are the consequences of detection errors?
8. What action should the company take next?

## Limitations

PaySim is synthetic. Results do not establish real-world deployment
effectiveness. Transaction value is not automatically equivalent
to actual financial loss.