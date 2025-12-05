var socket = io();
var currentStart = null;
var currentEnd = null;

// --- DOM ELEMENTS ---
var plotDiv = document.getElementById('chart');
var btnLeft = document.getElementById('pan_left');
var btnRight = document.getElementById('pan_right');
var roiLabel = document.getElementById('roi_val');
var winLabel = document.getElementById('win_val');
var countLabel = document.getElementById('trade_count');

// --- BUTTON LISTENERS ---
btnLeft.addEventListener('click', function() {
    shiftTime(-30); // Back 30 days
});

btnRight.addEventListener('click', function() {
    shiftTime(30); // Fwd 30 days
});

function shiftTime(days) {
    if (!currentStart || !currentEnd) return;
    
    // Parse current strings to Date objects
    let s = new Date(currentStart);
    let e = new Date(currentEnd);
    
    // Add Days
    s.setDate(s.getDate() + days);
    e.setDate(e.getDate() + days);
    
    // Emit ISO Strings
    socket.emit('range_change', {
        start: s.toISOString(),
        end: e.toISOString()
    });
}

// --- SOCKET UPDATE LISTENER ---
socket.on('update_traces', function(data) {
    
    // 1. Update HTML Stats
    roiLabel.innerText = "$" + data.roi.toFixed(0);
    roiLabel.className = data.roi >= 0 ? 'pos' : 'neg';
    winLabel.innerText = data.win.toFixed(1) + "%";
    countLabel.innerText = data.count;

    // 2. Update Global Range Variables (for panning)
    if(data.layout.xaxis && data.layout.xaxis.range) {
        currentStart = data.layout.xaxis.range[0];
        currentEnd = data.layout.xaxis.range[1];
    }

    // 3. Render Chart
    Plotly.react(plotDiv, data.traces, data.layout, {responsive: true});

    // 4. Re-attach interactions
    attachInteractions();
});

// --- CHART INTERACTION LOGIC ---
function attachInteractions() {
    plotDiv.removeAllListeners('plotly_relayout');
    plotDiv.removeAllListeners('plotly_hover');
    plotDiv.removeAllListeners('plotly_unhover');

    // Detect Zoom/Pan via Mouse
    plotDiv.on('plotly_relayout', function(data) {
        // Plotly sends different keys depending on interaction
        let s, e;
        if(data['xaxis.range[0]']) {
            s = data['xaxis.range[0]'];
            e = data['xaxis.range[1]'];
        } else if (data['xaxis.range']) {
            s = data['xaxis.range'][0];
            e = data['xaxis.range'][1];
        }

        // Only emit if we have valid dates
        if (s && e) {
            socket.emit('range_change', {start: s, end: e});
        }
    });

    // Vertical Crosshair Logic
    // Trace Indices correspond to the last 4 traces added in Python (Hover Lines)
    // 0:Price, 1:EMA, 2:Long, 3:Short, 4:Win, 5:Loss, 6:Exit, 7:MACD, 8:Sig, 9:Hist, 10:Daily, 11:Equity
    // 12: HoverY1, 13: HoverY2, 14: HoverY3, 15: HoverY4
    var hoverIndices = [12, 13, 14, 15];

    plotDiv.on('plotly_hover', function(data) {
        if (!data.points || data.points.length === 0) return;
        
        var xVal = data.points[0].x;
        
        // Define ranges for the vertical lines based on current layout axes
        var updates = [];
        var yAxes = ['yaxis', 'yaxis2', 'yaxis3', 'yaxis4'];
        
        for (var i = 0; i < hoverIndices.length; i++) {
            var axis = yAxes[i];
            var range = plotDiv.layout[axis].range;
            
            // Safety check if range exists
            if (!range) range = [0, 1];

            updates.push({
                x: [[xVal, xVal]], 
                y: [[range[0], range[1]]] 
            });
        }

        // Efficiently update only the hover traces
        Plotly.update(plotDiv, {
            x: updates.map(u => u.x[0]),
            y: updates.map(u => u.y[0])
        }, {}, hoverIndices);
    });

    plotDiv.on('plotly_unhover', function(data) {
        // Clear lines
        var empty = [[],[],[],[]];
        Plotly.update(plotDiv, {x: empty, y: empty}, {}, hoverIndices);
    });
}