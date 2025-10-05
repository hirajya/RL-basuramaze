class BaseAgent {
    constructor(params) {
        if (!params || typeof params !== 'object') {
            throw new Error('Invalid parameters provided to agent');
        }
        this.learningRate = params.learningRate || 0.1;
        this.gamma = params.gamma || 0.99;
        this.epsilon = params.epsilon || 0.1;
        this.actionSpace = 4; // up, right, down, left
        this.rewardHistory = [];
    }

    // Convert state object to string for table lookup
    stateToString(state) {
        if (!state || !state.wallE || !state.evilRobot || typeof state.trashCount !== 'number') {
            console.error('Invalid state:', state);
            return '0,0,0,0,0'; // Default state string
        }
        return `${state.wallE.x},${state.wallE.y},${state.evilRobot.x},${state.evilRobot.y},${state.trashCount}`;
    }

    // Epsilon-greedy action selection
    epsilonGreedy(state, values) {
        if (!state || !values) {
            console.error('Invalid state or values for epsilon-greedy selection');
            return Math.floor(Math.random() * this.actionSpace); // Random action as fallback
        }

        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.actionSpace);
        }
        return this.getBestAction(state, values);
    }

    // Get best action based on values
    getBestAction(state, values) {
        try {
            const stateStr = this.stateToString(state);
            let maxValue = -Infinity;
            let bestAction = 0;

            for (let action = 0; action < this.actionSpace; action++) {
                const value = values[stateStr]?.[action] || 0;
                if (value > maxValue) {
                    maxValue = value;
                    bestAction = action;
                }
            }
            
            return bestAction;
        } catch (error) {
            console.error('Error in getBestAction:', error);
            return Math.floor(Math.random() * this.actionSpace); // Random action as fallback
        }
    }

    // Update parameters
    updateParams(params) {
        if (!params) return;
        if (typeof params.learningRate === 'number') this.learningRate = params.learningRate;
        if (typeof params.gamma === 'number') this.gamma = params.gamma;
        if (typeof params.epsilon === 'number') this.epsilon = params.epsilon;
    }

    // Get average reward over last N episodes
    getAverageReward(n = 10) {
        if (this.rewardHistory.length === 0) return 0;
        const lastN = this.rewardHistory.slice(-Math.min(n, this.rewardHistory.length));
        return lastN.reduce((a, b) => a + b, 0) / lastN.length;
    }

    // Add episode reward to history
    addReward(totalReward) {
        if (typeof totalReward === 'number') {
            this.rewardHistory.push(totalReward);
        }
    }

    // Reset agent
    reset() {
        this.rewardHistory = [];
    }
}