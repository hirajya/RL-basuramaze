class MonteCarloAgent extends BaseAgent {
    constructor(params) {
        super(params);
        this.qTable = {};
        this.episodeStates = [];
        this.episodeActions = [];
        this.episodeRewards = [];
        this.episodeCount = 0;
        
        // Track episode transitions for table export
        this.episodeTransitions = [];
        this.currentEpisode = 0;
        this.returnHistory = []; // Track returns for export with episode reset column
    }

    // Set current episode number (called from simulation)
    setEpisode(episodeNum) {
        this.currentEpisode = episodeNum;
        // Add simple episode reset marker to return history
        this.returnHistory.push({
            episode: episodeNum,
            step: 'RESET',
            state: `Episode_${episodeNum}_Start`,
            action: 'RESET',
            returnValue: '0.0000',
            reward: 'WALL-E_RESET',
            timestamp: new Date().toISOString(),
            episodeReset: 'TRUE'  // Clear indicator for episode reset
        });
        // Add episode start marker
        this.episodeTransitions.push({
            episode: episodeNum,
            marker: `EPISODE_${episodeNum}_START`,
            timestamp: new Date().toISOString(),
            type: 'START'
        });
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
        // Add episode end marker before processing
        if (this.currentEpisode > 0) {
            this.episodeTransitions.push({
                episode: this.currentEpisode,
                marker: `EPISODE_${this.currentEpisode}_END`,
                timestamp: new Date().toISOString(),
                type: 'END'
            });
        }

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

            // Update Q-values and track for export
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
                
                // Track this return update for export
                this.returnHistory.push({
                    episode: this.currentEpisode,
                    step: t + 1,
                    state: stateStr,
                    action: action,
                    returnValue: returns[t].toFixed(4),
                    reward: this.episodeRewards[t],
                    timestamp: new Date().toISOString(),
                    episodeReset: 'FALSE'  // Normal update, not a reset
                });
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
        this.episodeTransitions = [];
        this.currentEpisode = 0;
        this.returnHistory = []; // Reset return history
    }

    // Export state-action return table with episode reset column
    exportReturnTable() {
        // Export return history with Episode_Reset column
        const csvData = ['Episode,Step,State,Action,Return_Value,Reward,Episode_Reset,Timestamp'];
        
        this.returnHistory.forEach(entry => {
            csvData.push(`${entry.episode},${entry.step},${entry.state},${entry.action},${entry.returnValue},${entry.reward || ''},${entry.episodeReset || 'FALSE'},${entry.timestamp}`);
        });
        
        const csvContent = csvData.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `monte_carlo_returns_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        
        URL.revokeObjectURL(url);

        // ...rest of existing returnData logic...
        const returnData = [];
        const actionNames = ['Up', 'Right', 'Down', 'Left'];
        
        // Add episode transition markers at the beginning
        for (const transition of this.episodeTransitions) {
            returnData.push({
                state: transition.marker,
                wallE_x: `--- ${transition.type} ---`,
                wallE_y: `Episode ${transition.episode}`,
                evil_x: '---',
                evil_y: '---',
                trashCount: '---',
                action: '---',
                actionName: transition.marker,
                returnValue: '0.0000',
                isOptimalAction: '---',
                episodeMarker: transition.marker,
                timestamp: transition.timestamp
            });
        }
        
        // Only export states that were actually visited (have non-zero values)
        for (const [stateStr, qValues] of Object.entries(this.qTable)) {
            const stateParts = stateStr.split(',');
            if (stateParts.length >= 5) {
                const wallE_x = stateParts[0];
                const wallE_y = stateParts[1];
                const evil_x = stateParts[2];
                const evil_y = stateParts[3];
                const trashCount = stateParts[4];
                
                // Only add rows for actions that have been updated (non-zero values)
                for (let action = 0; action < qValues.length; action++) {
                    if (Math.abs(qValues[action]) > 0.001) { // Only export meaningful values
                        returnData.push({
                            state: stateStr,
                            wallE_x: wallE_x,
                            wallE_y: wallE_y,
                            evil_x: evil_x,
                            evil_y: evil_y,
                            trashCount: trashCount,
                            action: action,
                            actionName: actionNames[action],
                            returnValue: qValues[action].toFixed(4),
                            isOptimalAction: qValues[action] === Math.max(...qValues) ? 'TRUE' : 'FALSE',
                            episodeMarker: '',
                            timestamp: ''
                        });
                    }
                }
            }
        }
        
        return returnData;
    }

    // Export policy table with episode transition markers
    exportPolicyTable() {
        const policyData = [];
        const actionNames = ['Up', 'Right', 'Down', 'Left'];
        
        // Add episode transition markers at the beginning
        for (const transition of this.episodeTransitions) {
            policyData.push({
                state: transition.marker,
                wallE_x: `--- ${transition.type} ---`,
                wallE_y: `Episode ${transition.episode}`,
                evil_x: '---',
                evil_y: '---',
                trashCount: '---',
                optimalAction: '---',
                optimalActionName: transition.marker,
                returnValue: '0.0000',
                episodeMarker: transition.marker,
                timestamp: transition.timestamp
            });
        }
        
        // Only export states that were actually visited
        for (const [stateStr, qValues] of Object.entries(this.qTable)) {
            const stateParts = stateStr.split(',');
            if (stateParts.length >= 5) {
                // Check if this state has meaningful values
                if (qValues.some(q => Math.abs(q) > 0.001)) {
                    const maxValue = Math.max(...qValues);
                    const optimalAction = qValues.indexOf(maxValue);
                    
                    policyData.push({
                        state: stateStr,
                        wallE_x: stateParts[0],
                        wallE_y: stateParts[1],
                        evil_x: stateParts[2],
                        evil_y: stateParts[3],
                        trashCount: stateParts[4],
                        optimalAction: optimalAction,
                        optimalActionName: actionNames[optimalAction],
                        returnValue: maxValue.toFixed(4),
                        episodeMarker: '',
                        timestamp: ''
                    });
                }
            }
        }
        
        return policyData;
    }

    // Get return table statistics
    getReturnTableStats() {
        const states = Object.keys(this.qTable).length;
        const totalEntries = states * 4; // 4 actions per state
        const allReturns = Object.values(this.qTable).flat();
        const nonZeroEntries = allReturns.filter(r => Math.abs(r) > 0.001).length;
        
        return {
            totalStates: states,
            totalEntries: totalEntries,
            nonZeroEntries: nonZeroEntries,
            coverage: totalEntries > 0 ? (nonZeroEntries / totalEntries * 100).toFixed(1) + '%' : '0%',
            minReturn: allReturns.length > 0 ? Math.min(...allReturns).toFixed(4) : '0',
            maxReturn: allReturns.length > 0 ? Math.max(...allReturns).toFixed(4) : '0',
            avgReturn: allReturns.length > 0 ? (allReturns.reduce((a, b) => a + b, 0) / allReturns.length).toFixed(4) : '0'
        };
    }
}