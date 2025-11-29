#Jeremiah Emery
#Prof. Omar El Khatib
#Artificial Intelligence 


import random

# ---------------- Environment ----------------

states = []
for x in range(3):
    for y in range(4):
        states.append((x, y))

actions = ['U', 'D', 'L', 'R']
terminal_states = [(1, 1), (1, 2), (2, 1), (2, 3)]

rewards = {
    (1, 1): -10,
    (2, 1): -20,
    (1, 2): 10,
    (2, 3): 20,
}

# reward 0 for all other states
for s in states:
    if s not in rewards:
        rewards[s] = 0

# for printing small policy paths in the summary table
ARROWS = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}

# pick two reference states for V(2) and V(3) columns in the table
STATE_2 = (0, 0)
STATE_3 = (1, 0)

notes_by_gamma = {
    0.9: "Long-term planning",
    0.5: "Moderate-term planning",
    0.1: "Short-sighted planning"
}


def apply(s, a):
    """Deterministic transition s -> s1 under action a."""
    if s in terminal_states:
        return s
    x, y = s
    if a == 'U':
        y = min(y + 1, 3)
    elif a == 'D':
        y = max(y - 1, 0)
    elif a == 'L':
        x = max(x - 1, 0)
    elif a == 'R':
        x = min(x + 1, 2)
    return (x, y)


def print_values(V, title):
    print("==", title, "==")
    for y in reversed(range(4)):
        row = []
        for x in range(3):
            v = V.get((x, y), 0.0)
            row.append(f"{v:6.2f}")
        print("y =", y, "|", " ".join(row))
    print()


def print_policy(policy, title):
    print("==", title, "==")
    for y in reversed(range(4)):
        row = []
        for x in range(3):
            s = (x, y)
            if s in terminal_states:
                row.append(" T ")
            else:
                a = policy.get(s)
                if a is None:
                    row.append(" . ")
                else:
                    row.append(" " + a + " ")
        print("y =", y, "|", " ".join(row))
    print()


def policy_path(start_state, policy, max_steps=6):
    """Short path as [→,→,↑,...] starting from start_state."""
    s = start_state
    path = []
    for _ in range(max_steps):
        a = policy.get(s)
        if a is None:
            break
        path.append(ARROWS.get(a, a))
        s1 = apply(s, a)
        if s1 in terminal_states:
            s = s1
            break
        s = s1
    return "[" + ",".join(path) + "]"


# ---------------- Value Iteration ----------------

def value_iteration(gamma, Max_Iterations=1000, theta=1e-6):
    V = {}
    for s in states:
        V[s] = 0.0

    iterations = 0
    for k in range(Max_Iterations):
        change = False
        for s in states:
            if s in terminal_states:
                continue
            v_old = V[s]
            best_v = float('-inf')
            for a in actions:
                s1 = apply(s, a)
                r = rewards[s1]
                if s1 in terminal_states:
                    v = r
                else:
                    v = r + gamma * V[s1]
                if v > best_v:
                    best_v = v
            V[s] = best_v
            if abs(v_old - V[s]) > theta:
                change = True
        iterations += 1
        if not change:
            break

    # Generate greedy policy:
    policy = {}
    for s in states:
        policy[s] = None
    for s in states:
        if s in terminal_states:
            continue
        best_v, best_a = float('-inf'), None
        for a in actions:
            s1 = apply(s, a)
            r = rewards[s1]
            if s1 in terminal_states:
                v = r
            else:
                v = r + gamma * V[s1]
            if v > best_v:
                best_v, best_a = v, a
        policy[s] = best_a

    return V, policy, iterations


# ---------------- Policy Iteration ----------------

def policy_iteration(gamma, Max_Iterations=1000, theta=1e-6):
    policy = {}
    for s in states:
        if s in terminal_states:
            policy[s] = None
        else:
            policy[s] = random.choice(actions)  # similar to prof's random init

    iterations = 0
    while True:
        # Policy Evaluation:
        V = {}
        for s in states:
            V[s] = 0.0

        for value_loop in range(Max_Iterations):
            change = False
            for s in states:
                if s in terminal_states:
                    continue
                v_old = V[s]
                a = policy[s]
                s1 = apply(s, a)
                r = rewards[s1]
                if s1 in terminal_states:
                    V[s] = r
                else:
                    V[s] = r + gamma * V[s1]
                if abs(v_old - V[s]) > theta:
                    change = True
            if not change:
                break

        # Improve policy:
        policy_stable = True
        for s in states:
            if s in terminal_states:
                continue
            old_a = policy[s]
            best_v, best_a = float('-inf'), None
            for a in actions:
                s1 = apply(s, a)
                r = rewards[s1]
                if s1 in terminal_states:
                    v = r
                else:
                    v = r + gamma * V[s1]
                if v > best_v:
                    best_v, best_a = v, a
            if best_a != old_a:
                policy[s] = best_a
                policy_stable = False

        iterations += 1
        if policy_stable or iterations >= Max_Iterations:
            break

    return V, policy, iterations


# ---------------- Q-Learning ----------------

def q_learning(gamma, alpha=0.1, epsilon=0.1,
               Max_Episodes=30000, tol=1e-4):
    Q = {}
    for s in states:
        for a in actions:
            Q[(s, a)] = 0.0

    last_max_change = 0.0
    episodes_used = 0

    for ep in range(1, Max_Episodes + 1):
        # start from a random non-terminal state
        s = random.choice([st for st in states if st not in terminal_states])

        while True:
            # epsilon-greedy action
            if random.random() < epsilon:
                a = random.choice(actions)
            else:
                best_q, best_a = float('-inf'), None
                for act in actions:
                    q = Q[(s, act)]
                    if q > best_q:
                        best_q, best_a = q, act
                a = best_a

            s1 = apply(s, a)
            r = rewards[s1]

            if s1 in terminal_states:
                max_next = 0.0
            else:
                max_next = max(Q[(s1, act)] for act in actions)

            target = r + gamma * max_next
            old_q = Q[(s, a)]
            Q[(s, a)] = old_q + alpha * (target - old_q)
            change = abs(Q[(s, a)] - old_q)
            if change > last_max_change:
                last_max_change = change

            s = s1
            if s1 in terminal_states:
                break

        episodes_used = ep
        if ep % 1000 == 0:
            if last_max_change < tol:
                break
            last_max_change = 0.0

    # Build V from Q and greedy policy:
    V = {}
    policy = {}
    for s in states:
        if s in terminal_states:
            V[s] = 0.0
            policy[s] = None
            continue
        best_q, best_a = float('-inf'), None
        for a in actions:
            q = Q[(s, a)]
            if q > best_q:
                best_q, best_a = q, a
        V[s] = best_q
        policy[s] = best_a

    return V, policy, episodes_used


# ---------------- Summary table ----------------

summary_rows = []


def add_summary(algorithm, gamma, iteration, V, policy):
    v2 = V.get(STATE_2, 0.0)
    v3 = V.get(STATE_3, 0.0)
    path = policy_path(STATE_2, policy, max_steps=6)
    notes = notes_by_gamma.get(gamma, "")
    row = {
        "Algorithm": algorithm,
        "gamma": gamma,
        "Iteration": iteration,
        "Policy": path,
        "V2": v2,
        "V3": v3,
        "Notes": notes
    }
    summary_rows.append(row)


def print_summary_table():
    print("=" * 80)
    print("Summary Table (suggested format style)")
    print("=" * 80)
    header = f"{'Algorithm':15} {'γ':5} {'Iteration':10} {'Policy':20} {'V(2)':8} {'V(3)':8} {'Notes'}"
    print(header)
    print("-" * 80)
    for row in summary_rows:
        print(f"{row['Algorithm']:15} "
              f"{row['gamma']:<5} "
              f"{row['Iteration']:<10} "
              f"{row['Policy']:20} "
              f"{row['V2']:8.2f} "
              f"{row['V3']:8.2f} "
              f"{row['Notes']}")
    print("=" * 80)


# ---------------- Main ----------------

if __name__ == "__main__":
    gammas = [0.9, 0.5, 0.1]

    for gamma in gammas:
        print("#" * 60)
        print("gamma =", gamma)
        print("#" * 60)

        # Value Iteration
        V_vi, policy_vi, it_vi = value_iteration(gamma)
        print("Value Iteration - iterations:", it_vi)
        print_values(V_vi, "V(s) from Value Iteration, gamma=" + str(gamma))
        print_policy(policy_vi, "Policy from Value Iteration, gamma=" + str(gamma))
        add_summary("Value Iteration", gamma, it_vi, V_vi, policy_vi)

        # Policy Iteration
        V_pi, policy_pi, it_pi = policy_iteration(gamma)
        print("Policy Iteration - iterations:", it_pi)
        print_values(V_pi, "V(s) from Policy Iteration, gamma=" + str(gamma))
        print_policy(policy_pi, "Policy from Policy Iteration, gamma=" + str(gamma))
        add_summary("Policy Iteration", gamma, it_pi, V_pi, policy_pi)

        # Q-Learning
        V_ql, policy_ql, it_ql = q_learning(gamma)
        print("Q-Learning - episodes:", it_ql)
        print_values(V_ql, "Approx. V(s) from Q-Learning, gamma=" + str(gamma))
        print_policy(policy_ql, "Policy from Q-Learning, gamma=" + str(gamma))
        add_summary("Q-Learning", gamma, it_ql, V_ql, policy_ql)

    print_summary_table()
