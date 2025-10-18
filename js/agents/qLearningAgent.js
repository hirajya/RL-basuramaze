class QLearningAgent extends BaseAgent {
    constructor(params) {
        super(params);
        this.qTable = {};
        this.minQValue = -20;
        this.maxQValue = 50;
        
        // Track Q-table updates for export
        this.tableHistory = [];
        this.currentEpisode = 0;
        this.stepCount = 0;
    }

    // Set current episode number (called from simulation)
    setEpisode(episodeNum) {
        this.currentEpisode = episodeNum;
        this.stepCount = 0;
        
        // Mark the first entry of each episode as a reset
        this.isEpisodeStart = true;
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

            this.stepCount++;

            // Track this update for export with Episode_Reset flag
            this.tableHistory.push({
                state: stateStr,
                wallEX: state.wallE.x,
                wallEY: state.wallE.y,
                evilRobotX: state.evilRobot.x,
                evilRobotY: state.evilRobot.y,
                trashCount: state.trashCount,
                action: action,
                actionName: this.getActionName(action),
                qValue: newQ,
                isOptimal: this.qTable[stateStr][action] === Math.max(...this.qTable[stateStr]),
                episodeReset: this.isEpisodeStart || false
            });

            // Clear episode start flag after first update
            this.isEpisodeStart = false;

            // Normalize Q-values if they get too large
            if (Math.abs(this.maxQValue) > 1000 || Math.abs(this.minQValue) > 1000) {
                this.normalizeQValues();
            }

        } catch (error) {
            console.error('Error in Q-Learning update:', error);
        }
    }

    getActionName(action) {
        const actionNames = ['Up', 'Right', 'Down', 'Left'];
        return actionNames[action] || 'Unknown';
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
        // Reset Q-table and related data structures
        this.qTable = {};
        this.minQValue = -20;
        this.maxQValue = 50;
        
        // Reset tracking data for exports
        this.tableHistory = [];
        this.currentEpisode = 0;
        this.stepCount = 0;
        
        // Reset episode flags
        this.isEpisodeStart = false;
        
        console.log('Q-Learning agent reset: Q-table and history cleared');
    }

    // Export Q-table data for the main simulation export
    exportQTable() {
        const qtableData = [];
        const actionNames = ['Up', 'Right', 'Down', 'Left'];
        
        // Only export states that have been actually visited (non-zero Q-values)
        for (const [stateStr, qValues] of Object.entries(this.qTable)) {
            const stateParts = stateStr.split(',');
            if (stateParts.length >= 5) {
                const wallE_x = parseInt(stateParts[0]);
                const wallE_y = parseInt(stateParts[1]);
                const evil_x = parseInt(stateParts[2]);
                const evil_y = parseInt(stateParts[3]);
                const trashCount = parseInt(stateParts[4]);
                
                // Find the optimal action for this state
                const maxQValue = Math.max(...qValues);
                const optimalAction = qValues.indexOf(maxQValue);
                
                // Only add rows for actions that have been updated (non-zero values)
                for (let action = 0; action < qValues.length; action++) {
                    if (Math.abs(qValues[action]) > 0.001) { // Only export meaningful values
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
                            isOptimalAction: action === optimalAction ? 'TRUE' : 'FALSE'
                        });
                    }
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