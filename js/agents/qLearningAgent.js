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
        this.qTable = {};
        this.minQValue = -20;
        this.maxQValue = 50;
        this.episodeTransitions = [];
        this.currentEpisode = 0;
        this.tableHistory = []; // Reset Q-table history
    }

    // Export Q-table in the exact format shown in CSV
    exportQTable() {
        const csvData = ['State,Wall-E X,Wall-E Y,Evil Robot,Evil Robot,Trash Cou,Action ID,Action Na,Q-Value,Is Optima,EPISODE'];
        
        this.tableHistory.forEach((entry, index) => {
            csvData.push([
                entry.state,
                entry.wallEX,
                entry.wallEY,
                entry.evilRobotX,
                entry.evilRobotY,
                entry.trashCount,
                entry.action,
                entry.actionName,
                entry.qValue.toFixed(4),
                entry.isOptimal ? 'TRUE' : 'FALSE',
                entry.episodeReset ? 'TRUE' : 'FALSE'
            ].join(','));
        });
        
        const csvContent = csvData.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `q_learning_qtable_export_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        
        URL.revokeObjectURL(url);
        
        return this.tableHistory;
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