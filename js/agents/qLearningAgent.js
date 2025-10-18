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

    // Export Q-table with state-action values
    exportQTable() {
        const qtableData = [];
        const actionNames = ['Up', 'Right', 'Down', 'Left'];
        
        // Convert Q-table to exportable format
        for (const [stateStr, qValues] of Object.entries(this.qTable)) {
            // Parse state string back to components
            const stateParts = stateStr.split(',');
            if (stateParts.length >= 5) {
                const wallE_x = stateParts[0];
                const wallE_y = stateParts[1];
                const evil_x = stateParts[2];
                const evil_y = stateParts[3];
                const trashCount = stateParts[4];
                
                // Add a row for each action
                for (let action = 0; action < qValues.length; action++) {
                    qtableData.push({
                        state: stateStr,
                        wallE_x: wallE_x,
                        wallE_y: wallE_y,
                        evil_x: evil_x,
                        evil_y: evil_y,
                        trashCount: trashCount,
                        action: action,
                        actionName: actionNames[action],
                        qValue: qValues[action].toFixed(4),
                        isOptimalAction: qValues[action] === Math.max(...qValues) ? 'TRUE' : 'FALSE'
                    });
                }
            }
        }
        
        return qtableData;
    }

    // Get Q-table statistics
    getQTableStats() {
        const states = Object.keys(this.qTable).length;
        const totalEntries = states * 4; // 4 actions per state
        const nonZeroEntries = Object.values(this.qTable)
            .flat()
            .filter(q => Math.abs(q) > 0.001).length;
        
        return {
            totalStates: states,
            totalEntries: totalEntries,
            nonZeroEntries: nonZeroEntries,
            coverage: totalEntries > 0 ? (nonZeroEntries / totalEntries * 100).toFixed(1) + '%' : '0%',
            minQValue: this.minQValue.toFixed(4),
            maxQValue: this.maxQValue.toFixed(4)
        };
    }
}