class MonteCarloAgent extends BaseAgent {
    constructor(params) {
        super(params);
        this.qTable = {};
        this.episodeStates = [];
        this.episodeActions = [];
        this.episodeRewards = [];
        this.episodeCount = 0;
    }

    selectAction(state) {
        // Update state values in environment
        const stateStr = this.stateToString(state);
        if (this.qTable[stateStr]) {
            // Use max Q-value as state value
            const stateValue = Math.max(...this.qTable[stateStr]);
            window.env.updateStateValue(state.wallE.x, state.wallE.y, stateValue);
        }
        try {
            const action = this.epsilonGreedy(state, this.qTable);
            return action;
        } catch (error) {
            console.error('Error in selectAction:', error);
            return Math.floor(Math.random() * this.actionSpace);
        }
    }

    update(state, action, reward) {
        if (!state || !Number.isInteger(action) || typeof reward !== 'number') {
            console.error('Invalid input to update:', { state, action, reward });
            return;
        }
        
        try {
            // Store state-action-reward for the episode
            const stateStr = this.stateToString(state);
            this.episodeStates.push(stateStr);
            this.episodeActions.push(action);
            this.episodeRewards.push(reward);
        } catch (error) {
            console.error('Error in update:', error);
        }
    }

    episodeEnd() {
        if (this.episodeStates.length === 0) {
            console.warn('Episode ended with no data');
            return;
        }

        try {
            // Calculate returns for each step
            let G = 0;
            const returns = [];
            
            // Calculate returns in reverse order
            for (let t = this.episodeRewards.length - 1; t >= 0; t--) {
                G = this.episodeRewards[t] + this.gamma * G;
                returns.unshift(G);
            }

            // Update Q-values for each state-action pair
            for (let t = 0; t < this.episodeStates.length; t++) {
                const stateStr = this.episodeStates[t];
                const action = this.episodeActions[t];
                
                // Initialize state in Q-table if not exists
                if (!this.qTable[stateStr]) {
                    this.qTable[stateStr] = Array(this.actionSpace).fill(0);
                }
                
                // Update Q-value using incremental mean with learning rate
                const oldQ = this.qTable[stateStr][action];
                this.qTable[stateStr][action] = oldQ + this.learningRate * (returns[t] - oldQ);
            }

            // Store total episode reward
            const totalReward = this.episodeRewards.reduce((a, b) => a + b, 0);
            this.addReward(totalReward);
            
            // Log episode stats
            this.episodeCount++;

            // Clear episode memory
            this.episodeStates = [];
            this.episodeActions = [];
            this.episodeRewards = [];
        } catch (error) {
            console.error('Error in episodeEnd:', error);
            // Reset episode data on error
            this.episodeStates = [];
            this.episodeActions = [];
            this.episodeRewards = [];
        }
    }

    reset() {
        super.reset();
        this.qTable = {};
        this.episodeStates = [];
        this.episodeActions = [];
        this.episodeRewards = [];
        this.episodeCount = 0;
    }
}