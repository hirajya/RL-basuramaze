class QLearningAgent extends BaseAgent {
    constructor(params) {
        super(params);
        this.qTable = {};
        this.minQValue = -20;
        this.maxQValue = 50;
    }

    selectAction(state) {
        if (!state || !state.wallE) {
            console.error('Invalid state in selectAction');
            return Math.floor(Math.random() * this.actionSpace);
        }

        const action = this.epsilonGreedy(state, this.qTable);
        
        // Update state values in environment
        const stateStr = this.stateToString(state);
        if (this.qTable[stateStr]) {
            // Use normalized Q-value as state value
            const stateValue = Math.max(...this.qTable[stateStr]);
            window.env.updateStateValue(state.wallE.x, state.wallE.y, stateValue);
        }
        
        return action;
    }

    update(state, action, reward, nextState) {
        if (!state || !nextState || !Number.isInteger(action) || typeof reward !== 'number') {
            console.error('Invalid input to update:', { state, action, reward, nextState });
            return;
        }

        try {
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
            const newQ = currentQ + this.learningRate * (
                reward + this.gamma * maxNextQ - currentQ
            );

            // Update Q-value and track min/max for normalization
            this.qTable[stateStr][action] = newQ;
            this.minQValue = Math.min(this.minQValue, newQ);
            this.maxQValue = Math.max(this.maxQValue, newQ);

            // Normalize Q-values if they get too large
            if (Math.abs(this.maxQValue) > 1000 || Math.abs(this.minQValue) > 1000) {
                this.normalizeQValues();
            }
        } catch (error) {
            console.error('Error in Q-Learning update:', error);
        }
    }

    normalizeQValues() {
        const range = this.maxQValue - this.minQValue;
        if (range === 0) return;

        for (const state in this.qTable) {
            this.qTable[state] = this.qTable[state].map(q => 
                (q - this.minQValue) / range * 70 - 20  // Scale to [-20, 50] range
            );
        }
        this.minQValue = -20;
        this.maxQValue = 50;
    }

    reset() {
        super.reset();
        this.qTable = {};
        this.minQValue = -20;
        this.maxQValue = 50;
    }
}