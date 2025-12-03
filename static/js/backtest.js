var socket = io();
var currentStart = null;
var currentEnd = null;

function attachListener() {
    var plotDiv = document.getElementById('chart');
    if (plotDiv) {
        plotDiv.on('plotly_relayout', function(data) {
            if (data['xaxis.range[0]'] && data['xaxis.range[1]']) {
                var newStart = data['xaxis.range[0]'];
                var newEnd = data['xaxis.range[1]'];
                if (newStart !== currentStart || newEnd !== currentEnd) {
                    currentStart = newStart;
                    currentEnd = newEnd;
                    socket.emit('range_change', {start: newStart, end: newEnd});
                }
            }
        });

        // Handle hover for vertical lines
        plotDiv.on('plotly_hover', function(data) {
            if (data.points.length > 0) {
                var x = data.points[0].x;
                var updates = [];
                var traceIndices = [6, 7, 8, 9]; // indices of vertical line traces
                var yaxes = ['yaxis', 'yaxis2', 'yaxis3', 'yaxis4'];
                for (var i = 0; i < traceIndices.length; i++) {
                    var yrange = plotDiv.layout[yaxes[i]].range || [0, 1];
                    updates.push({
                        x: [x, x],
                        y: yrange
                    });
                }
                Plotly.update(plotDiv, updates, {}, traceIndices);
            }
        });

        plotDiv.on('plotly_unhover', function(data) {
            var updates = [];
            var traceIndices = [6, 7, 8, 9];
            for (var i = 0; i < traceIndices.length; i++) {
                updates.push({
                    x: [],
                    y: []
                });
            }
            Plotly.update(plotDiv, updates, {}, traceIndices);
        });
    }
}

attachListener();

document.getElementById('pan_left').addEventListener('click', function() {
    socket.emit('pan', {direction: 'left'});
});

document.getElementById('pan_right').addEventListener('click', function() {
    socket.emit('pan', {direction: 'right'});
});

socket.on('update_traces', function(data) {
    var plotDiv = document.getElementById('chart');
    if (plotDiv) {
        Plotly.react(plotDiv, data.traces, data.layout);
    }
});