class QLearningAgent extends BaseAgent {
    constructor(params) {
        super(params);
        this.qTable = {};
    }

    selectAction(state) {
        const action = this.epsilonGreedy(state, this.qTable);
        
        // Update state values in environment
        const stateStr = this.stateToString(state);
        if (this.qTable[stateStr]) {
            // Use max Q-value as state value
            const stateValue = Math.max(...this.qTable[stateStr]);
            window.env.updateStateValue(state.wallE.x, state.wallE.y, stateValue);
        }
        
        return action;
    }

    update(state, action, reward, nextState) {
        const stateStr = this.stateToString(state);
        const nextStateStr = this.stateToString(nextState);

        // Initialize Q-values if not exists
        if (!this.qTable[stateStr]) {
            this.qTable[stateStr] = Array(this.actionSpace).fill(0);
        }
        if (!this.qTable[nextStateStr]) {
            this.qTable[nextStateStr] = Array(this.actionSpace).fill(0);
        }

        // Q-Learning update rule
        const maxNextQ = Math.max(...this.qTable[nextStateStr]);
        const currentQ = this.qTable[stateStr][action];
        this.qTable[stateStr][action] = currentQ + this.learningRate * (
            reward + this.gamma * maxNextQ - currentQ
        );
    }

    reset() {
        super.reset();
        this.qTable = {};
    }
}