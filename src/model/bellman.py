def belmann_equation(states, actions, transactions, rewards, discount_factor=0.99, theta=1e-5):
    """
      V(s): The valeu state sss, which represents the long-term reward of being state.
      R(s,a): The immediate reward received for taking action aaa in state sss
      Y: factor discount (between 0 and 1) that determines importance of future rewards compared to immediate rewards.
      P(s'|s,a): The probability  of transaction state s' from state sss by taking action a.
      discount_factor: float (gamma) valuing futures states relative to current ones.
      theta: float, accuracy threshold for convergence of the Bellman equation. (WHEN ALGORITHM STOPS)
      max_a: The optimal action that maximizes the expected value future rewards.

      V(s): the regime
      R(s,a): rebalacing for the stocks, signals to more options in determinates sectors.
      Y: We can use risk return to calculate this factors.
      P(s' | s,a): we can use the matrix transactions for the policies
    """

    v = {s: 0.0 for s in states}
    policy = {s: None for s in states}

    #algorithm iteration

    while True:
      delta = 0
      for s in states:
        v_old = v[s]
        max_q_value = float('-inf')
        best_action = None

        for a in actions:
          q_value = rewards.get((s, a), 0)
          expected_future = 0

          for next_states in states:
            prob = transactions.get((s, a, next_states), 0)
            expected_future += prob * (discount_factor * v[next_states]) #projection inference for the prob future value

        q_value += expected_future

        if q_value > max_q_value:
          max_q_value = q_value
          best_action = a

        v[s] = max_q_value
        policy[s] = best_action
        delta = max(delta, abs(v_old - v[s])) #verify if have lost value between to groups

      if delta < theta:
        break

    return v, policy
