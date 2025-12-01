document.addEventListener('DOMContentLoaded', () => {
  const symbolSelect = document.getElementById('symbol-select');
  const intervalSelect = document.getElementById('interval-select');
  const limitSelect = document.getElementById('limit-select');
  const statusDisplay = document.getElementById('status-display');
  const currentSymbolSpan = document.getElementById('current-symbol');
  const currentIntervalSpan = document.getElementById('current-interval');
  const dataPointsSpan = document.getElementById('data-points');
  const toggleBtn = document.getElementById('toggle-right-panel');
  const rightPanel = document.getElementById('right-panel');
  const emailModal = document.getElementById('email-modal');
  const modalEmailInput = document.getElementById('modal-email-input');
  const modalSendBtn = document.getElementById('modal-send-email-btn');

  const progressContainer = document.getElementById('progress-container');
  const progressPercent = document.getElementById('progress-percent');
  const progressBarFill = document.getElementById('progress-bar-fill');
  const progressLabel = document.getElementById('progress-label');

  const shapeStatusEl = document.getElementById('shape-properties-status');
  const shapeCoordinatesEl = document.getElementById('shape-coordinates');
  const shapeLineOptionsEl = document.getElementById('shape-line-options');
  const shapeSendEmailCheckbox = document.getElementById('shape-send-email-checkbox');
  const deleteShapeBtn = document.getElementById('delete-shape-btn');

  const xAxisMinEl = document.getElementById('x-axis-min');
  const xAxisMaxEl = document.getElementById('x-axis-max');

  // AI elements
  const aiSuggestionBtn = document.getElementById('ai-suggestion-btn');
  const aiSuggestionTextarea = document.getElementById('ai-suggestion-textarea');
  const useLocalOllamaCheckbox = document.getElementById('use-local-ollama-checkbox');
  const localOllamaModelDiv = document.getElementById('local-ollama-model-div');
  const localOllamaModelSelect = document.getElementById('local-ollama-model-select');
 
  async function fetchCurrentUser() {
    try {
      const resp = await fetch('/me');
      if (!resp.ok) return null;
      const data = await resp.json();
      return data.authenticated ? data : null;
    } catch (e) {
      console.error('Failed to fetch current user:', e);
      return null;
    }
  }

  function updateUserEmailDisplay(user) {
    const label = document.getElementById('user-email');
    if (!label) return;
    if (!user) {
      label.textContent = 'Not logged in';
    } else if (user.is_guest) {
      label.textContent = 'Guest user';
    } else if (user.email) {
      label.textContent = user.email;
    } else {
      label.textContent = 'Authenticated';
    }
  }

  // Toggle right panel
  if (toggleBtn && rightPanel) {
    toggleBtn.addEventListener('click', function() {
      rightPanel.classList.toggle('open');
    });

    document.addEventListener('click', function(event) {
      if (window.innerWidth <= 768) {
        if (!rightPanel.contains(event.target) && !toggleBtn.contains(event.target)) {
          rightPanel.classList.remove('open');
        }
      }
    });
  }

  let debounceTimeout;
  let loadDataTimeout;
  let currentSymbol = '';
  let currentInterval = '';
  let selectedShapeIndex = null;
  let hoveredShapeIndex = null;

  let socket = null;
  let currentUser = null;

  let currentLoadId = 0;
  let activeLoadId = 0;
  let lastDisplayedProgress = 0;

  let currentAbortController = null;
  let aiChunks = {};
  let finalChunks = {};
  let aiComplete = false;
  let currentLivePrice = null;
  
  // Track last user shapes state to avoid re-saving on live price updates
  let lastUserShapesHash = null;

  function setProgress(percent, message) {
    if (!progressContainer || !progressBarFill || !progressPercent) {
      return;
    }

    const numeric = Number(percent);
    let clamped = Math.max(
      0,
      Math.min(100, Number.isFinite(numeric) ? numeric : 0)
    );

    // Enforce monotonic progress for the current load
    if (clamped < lastDisplayedProgress) {
      clamped = lastDisplayedProgress;
    } else {
      lastDisplayedProgress = clamped;
    }

    progressBarFill.style.width = `${clamped}%`;
    progressPercent.textContent = `${clamped.toFixed(1)}%`;

    if (message && statusDisplay) {
      statusDisplay.textContent = message;
    }
  }

  function resetProgress() {
    lastDisplayedProgress = 0;
    setProgress(0, 'Ready');
  }

  function updateXAxisRangeDisplay() {
    const chartEl = document.getElementById('chart');
    if (!chartEl || !chartEl.layout || !chartEl.layout.xaxis || !xAxisMinEl || !xAxisMaxEl) {
      return;
    }

    const xRange = chartEl.layout.xaxis.range;
    if (Array.isArray(xRange) && xRange.length === 2) {
      const minDate = new Date(xRange[0]);
      const maxDate = new Date(xRange[1]);

      if (!isNaN(minDate.getTime()) && !isNaN(maxDate.getTime())) {
        xAxisMinEl.textContent = minDate.toISOString();
        xAxisMaxEl.textContent = maxDate.toISOString();
      } else {
        xAxisMinEl.textContent = '--';
        xAxisMaxEl.textContent = '--';
      }
    } else {
      xAxisMinEl.textContent = '--';
      xAxisMaxEl.textContent = '--';
    }
  }

  function mapStageToOverallProgress(stage, direction, pctStage) {
    const p = Math.max(0, Math.min(100, Number(pctStage) || 0));

    // Approximate mapping:
    // - backfill (fetch_start, backfill): 0-50%
    // - forward fill (fetch_end, forward): 50-90%
    // - complete: 100%
    if (stage === 'fetch_start' && direction === 'backfill') {
      return 0 + (p / 100) * 50; // 0-50
    }
    if (stage === 'fetch_end' && direction === 'forward') {
      return 50 + (p / 100) * 40; // 50-90
    }
    if (stage === 'complete') {
      return 100;
    }
    if (stage === 'error') {
      // show as "finished" but you can style differently if needed
      return Math.max(p, 100);
    }
    // 'start' or unknown stages: just use p as-is
    return p;
  }

  function initSocket() {
    if (typeof io === 'undefined') {
      console.warn('Socket.IO client library not loaded');
      return;
    }

    if (!currentUser) {
      console.warn('initSocket called without authenticated user');
      return;
    }

    if (socket) {
      try {
        socket.disconnect();
      } catch (e) {
        console.error('Error disconnecting existing socket:', e);
      }
    }

    socket = io();

    const joinRoom = () => {
      socket.emit('join_user_room');
      loadData();
    };

    if (socket.connected) {
      joinRoom();
    } else {
      socket.on('connect', () => {
        console.log('Socket.IO connected');
        joinRoom();
      });
    }

    socket.on('progress', (payload) => {
      if (payload.type === 'ai_progress') {
        setProgress(payload.progress, payload.message || 'AI generating...');
        if (payload.stage === 'complete') {
          setTimeout(() => {
            resetProgress();
          }, 2000);
        }
        return;
      }

      if (!payload || payload.type !== 'data_progress') {
        return;
      }

      // Require a loadId and ensure it matches the active one
      if (payload.loadId == null) {
        return;
      }
      const payloadLoadId = Number(payload.loadId);
      if (!Number.isFinite(payloadLoadId) || payloadLoadId !== activeLoadId) {
        return;
      }

      const rawPct =
        typeof payload.progress === 'number'
          ? payload.progress
          : Number(payload.progress || 0);

      const mappedPct = mapStageToOverallProgress(
        payload.stage,
        payload.direction,
        rawPct
      );

      const label = payload.stage ? payload.stage.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) : 'Loading data...';

      if (progressLabel) {
        progressLabel.innerHTML = `<strong>${label}:</strong>`;
      }

      setProgress(mappedPct, label);

      if (payload.stage === 'complete' || payload.stage === 'error') {
        // Give the user a moment to see 100%, then reset
        setTimeout(() => {
          resetProgress();
        }, 2000);
      }
    });

    socket.on('connect_error', (err) => {
      console.error('Socket.IO connection error:', err);
    });

    socket.on('ai_response', (data) => {
      if (data.partial) {
        if (data.is_final) {
          finalChunks[data.chunk_id] = data.partial;
        } else {
          aiChunks[data.chunk_id] = data.partial;
          if (!aiComplete) {
            let chunks = Object.keys(aiChunks).sort((a, b) => a - b).map(k => aiChunks[k]);
            let assembled = chunks.join('');
            aiSuggestionTextarea.value = assembled;
            aiSuggestionTextarea.scrollTop = aiSuggestionTextarea.scrollHeight;
          }
        }
      }
      if (data.complete) {
        aiComplete = true;
        let chunks = Object.keys(finalChunks).sort((a, b) => a - b).map(k => finalChunks[k]);
        let assembled = chunks.join('');
        aiSuggestionTextarea.value = assembled;
        aiSuggestionTextarea.scrollTop = aiSuggestionTextarea.scrollHeight;
      }
    });

    socket.on('ai_error', (data) => {
      aiSuggestionTextarea.value = `Error: ${data.error}`;
      aiSuggestionTextarea.scrollTop = aiSuggestionTextarea.scrollHeight;
    });

    socket.on('live_price', (data) => {
      // Ignore errors or data without price
      if (data.error || !data.price) {
        return;
      }
      
      const symbol = data.symbol;
      const price = data.price;
      const timestamp = data.timestamp;
      
      // Only update if this is for the currently displayed symbol
      if (symbol === currentSymbol) {
        //console.log(`[live_price] Received ${symbol}: ${price} at ${new Date(timestamp).toISOString()}`);
        
        currentLivePrice = {
          symbol: symbol,
          price: price,
          timestamp: timestamp
        };
        
        // Update the live price line on the chart
        updateLivePriceLine(price);
        
        // Update live price display in right panel
        const livePriceDisplay = document.getElementById('live-price-display');
        if (livePriceDisplay) {
          livePriceDisplay.textContent = `$${price.toFixed(2)}`;
        }
      }
    });
  }

  // Initialize shape properties UI
  if (shapeLineOptionsEl) {
    shapeLineOptionsEl.style.display = 'none';
  }
  if (shapeSendEmailCheckbox) {
    shapeSendEmailCheckbox.checked = false;
    shapeSendEmailCheckbox.disabled = true;
  }
  if (deleteShapeBtn) {
    deleteShapeBtn.disabled = true;
  }

  if (shapeSendEmailCheckbox) {
    shapeSendEmailCheckbox.addEventListener('change', () => {
      if (selectedShapeIndex === null) {
        return;
      }
      const chartEl = document.getElementById('chart');
      if (!chartEl || !chartEl.layout) {
        return;
      }
      const layoutShapes = chartEl.layout.shapes || [];
      const fullLayoutShapes =
        chartEl._fullLayout && Array.isArray(chartEl._fullLayout.shapes)
          ? chartEl._fullLayout.shapes
          : [];
 
      const targetShape =
        layoutShapes[selectedShapeIndex] || fullLayoutShapes[selectedShapeIndex];
      if (!targetShape || targetShape.type !== 'line') {
        return;
      }
 
      const newValue = !!shapeSendEmailCheckbox.checked;
      targetShape.sendEmailOnCross = newValue;
      if (layoutShapes[selectedShapeIndex]) {
        layoutShapes[selectedShapeIndex].sendEmailOnCross = newValue;
      }
      if (fullLayoutShapes[selectedShapeIndex]) {
        fullLayoutShapes[selectedShapeIndex].sendEmailOnCross = newValue;
      }
 
      // Persist updated drawings (backend may choose to ignore the extra field)
      saveDrawings(currentSymbol);
    });
  }
 
  // Delete currently selected shape via right-panel button
  if (deleteShapeBtn) {
    deleteShapeBtn.addEventListener('click', async () => {
      console.log('[deleteButton] Delete button clicked, selectedShapeIndex:', selectedShapeIndex);
      
      if (selectedShapeIndex === null) {
        console.log('[deleteButton] No shape selected, aborting delete');
        statusDisplay.textContent = 'No shape selected';
        return;
      }
 
      const chartEl = document.getElementById('chart');
      const gd = chartEl;
      if (!gd || !gd.layout || !gd._fullLayout) {
        console.log('[deleteButton] Chart not ready, aborting delete');
        statusDisplay.textContent = 'Chart not ready';
        return;
      }
 
      const layoutShapes = gd.layout.shapes || [];
      const fullLayoutShapes =
        Array.isArray(gd._fullLayout.shapes) ? gd._fullLayout.shapes : [];
      const shapes = layoutShapes.length ? layoutShapes : fullLayoutShapes;
 
      console.log('[deleteButton] Current shapes before delete:', shapes.length);
      console.log('[deleteButton] Selected shape index:', selectedShapeIndex);
 
      if (!Array.isArray(shapes) || !shapes[selectedShapeIndex]) {
        console.log('[deleteButton] Shapes array invalid or shape not found, aborting delete');
        statusDisplay.textContent = 'Shape not found';
        return;
      }

      // Confirm deletion with user
      const shapeToDelete = shapes[selectedShapeIndex];
      const shapeType = shapeToDelete.type || 'shape';
 
      // Clear selection first to avoid race conditions
      selectedShapeIndex = null;
      hoveredShapeIndex = null;
 
      try {
        // Update the chart immediately (remove the shape visually)
        const newShapes = shapes.slice();
        newShapes.splice(newShapes.indexOf(shapeToDelete), 1);
 
        console.log('[deleteButton] Updating chart with new shapes array...');
        gd.layout.shapes = newShapes;
        if (Array.isArray(gd._fullLayout.shapes)) {
          gd._fullLayout.shapes = newShapes;
        }
 
        updateShapePropertiesPanel();
 
        // Wait for Plotly to actually update the chart before saving
        console.log('[deleteButton] Waiting for Plotly to update chart...');
        await Plotly.relayout(gd, { shapes: newShapes });
        
        console.log('[deleteButton] Chart updated, now saving to server...');
        statusDisplay.textContent = 'Deleting shape...';
        
        // Only proceed with saving if we have a valid symbol
        if (!currentSymbol) {
          console.log('[deleteButton] No current symbol, cannot save to server');
          statusDisplay.textContent = 'Shape deleted locally';
          return;
        }
        
        // Save to server with retry logic
        let retries = 3;
        while (retries > 0) {
          try {
            await saveDrawings(currentSymbol);
            console.log('[deleteButton] Successfully saved deletion to server');
            statusDisplay.textContent = 'Shape deleted successfully';
            break;
          } catch (saveError) {
            console.error(`[deleteButton] Save attempt failed (${retries} retries left):`, saveError);
            retries--;
            if (retries > 0) {
              await new Promise(resolve => setTimeout(resolve, 1000));
            } else {
              statusDisplay.textContent = 'Shape deleted locally (server save failed)';
              console.error('[deleteButton] All save attempts failed');
            }
          }
        }
        
        // Refresh volume profiles after successful deletion
        console.log('[deleteButton] Refreshing volume profiles...');
        await refreshRectangleVolumeProfiles();
        
        console.log('[deleteButton] Delete operation completed successfully');
        
      } catch (err) {
        console.error('[deleteButton] Failed to delete shape via button:', err);
        statusDisplay.textContent = `Delete failed: ${err.message}`;
      }
    });
  }

  // Modal send email / guest
  modalSendBtn.addEventListener('click', async () => {
    const email = modalEmailInput.value.trim();
  
    // No email -> guest session
    if (!email) {
      try {
        const resp = await fetch('/guest-login', { method: 'POST' });
        if (resp.ok) {
          emailModal.style.display = 'none';
          currentUser = await fetchCurrentUser();
          updateUserEmailDisplay(currentUser);
          initSocket();
          await loadData();
        } else {
          alert('Failed to start guest session');
        }
      } catch (err) {
        alert('Error: ' + err.message);
      }
      return;
    }
  
    // Email provided -> start magic-link flow
    try {
      const resp = await fetch('/start-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });
      if (resp.ok) {
        alert('Check your email for a login link to complete sign in.');
      } else {
        let data = {};
        try {
          data = await resp.json();
        } catch (e) {}
        alert(data.error || 'Failed to start login');
      }
    } catch (err) {
      alert('Error: ' + err.message);
    }
  });

  // Modal guest button
  const modalGuestBtn = document.getElementById('modal-guest-btn');
  if (modalGuestBtn) {
    modalGuestBtn.addEventListener('click', async () => {
      try {
        const resp = await fetch('/guest-login', { method: 'POST' });
        if (resp.ok) {
          emailModal.style.display = 'none';
          currentUser = await fetchCurrentUser();
          updateUserEmailDisplay(currentUser);
          initSocket();
          await loadData();
        } else {
          alert('Failed to start guest session');
        }
      } catch (err) {
        alert('Error: ' + err.message);
      }
    });
  }

  // Check login and load initial chart using server-side session (/me)
  (async () => {
    try {
      currentUser = await fetchCurrentUser();
      if (currentUser) {
        updateUserEmailDisplay(currentUser);
        emailModal.style.display = 'none';
        initSocket();
        await loadData();
      } else {
        emailModal.style.display = 'block';
      }
    } catch (e) {
      console.error('Failed to initialise auth state:', e);
      emailModal.style.display = 'block';
    }
  })();

  // Reload data when symbol, interval, or limit changes
  function debouncedLoadData() {
    clearTimeout(loadDataTimeout);
    loadDataTimeout = setTimeout(loadData, 500);
  }
  symbolSelect.addEventListener('change', debouncedLoadData);
  intervalSelect.addEventListener('change', debouncedLoadData);
  limitSelect.addEventListener('change', debouncedLoadData);

  // Toolbar buttons
  document.getElementById('autoscale-btn').addEventListener('click', async () => {
    console.log('[DEBUG autoscale-btn] Clicked autoscale button, calculating manual ranges from data');
    const chartEl = document.getElementById('chart');
    if (!chartEl) {
      console.error('[DEBUG autoscale-btn] Chart element not found');
      return;
    }

    // Check if we have chart data to calculate ranges from
    if (!chartEl.data || !chartEl.data.length || !chartEl.data[0]) {
      console.error('[DEBUG autoscale-btn] No chart data available to calculate ranges');
      return;
    }

    const traceData = chartEl.data[0];
    if (!traceData.x || !traceData.x.length || !traceData.low || !traceData.high) {
      console.error('[DEBUG autoscale-btn] Trace data missing x or price data');
      return;
    }

    try {
      // Calculate X-axis range (time) from data
      const xValues = traceData.x.map(x => new Date(x).getTime()).filter(t => Number.isFinite(t));
      const xMin = Math.min(...xValues);
      const xMax = Math.max(...xValues);

      if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || xMin >= xMax) {
        console.error('[DEBUG autoscale-btn] Invalid X range calculated:', { xMin, xMax });
        return;
      }

      // Calculate Y-axis range (price) from OHLC data with padding
      const yValues = [...traceData.low, ...traceData.high].filter(y => Number.isFinite(y));
      const yMinBase = Math.min(...yValues);
      const yMaxBase = Math.max(...yValues);

      if (!Number.isFinite(yMinBase) || !Number.isFinite(yMaxBase) || yMinBase >= yMaxBase) {
        console.error('[DEBUG autoscale-btn] Invalid Y range calculated:', { yMinBase, yMaxBase });
        return;
      }

      // Add 5% padding to Y range for better visualization
      const yPadding = (yMaxBase - yMinBase) * 0.05;
      const yMin = Math.max(0, yMinBase - yPadding); // Don't go below 0 for prices
      const yMax = yMaxBase + yPadding;

      console.log('[DEBUG autoscale-btn] Calculated ranges:', {
        xMin: new Date(xMin).toISOString(),
        xMax: new Date(xMax).toISOString(),
        yMin,
        yMax,
        yPadding
      });

      // Set explicit ranges calculated from data
      await Plotly.relayout(chartEl, {
        'xaxis.range': [new Date(xMin).toISOString(), new Date(xMax).toISOString()],
        'yaxis.range': [yMin, yMax]
      });

      // Log range after autoscale
      if (chartEl.layout && chartEl.layout.xaxis) {
        console.log('[DEBUG autoscale-btn] Final xaxis range:', {
          range: chartEl.layout.xaxis.range,
          autorange: chartEl.layout.xaxis.autorange
        });
      }

      console.log('[DEBUG autoscale-btn] Manual autoscale applied successfully');
    } catch (err) {
      console.error('[DEBUG autoscale-btn] Error applying manual autoscale:', err);
    }
  });

  document.getElementById('pan-mode-btn').addEventListener('click', () => {
    Plotly.relayout('chart', {
      dragmode: 'pan'
    });
  });

  document.getElementById('zoom-in-btn').addEventListener('click', () => {
    const chartEl = document.getElementById('chart');
    if (!chartEl || !chartEl.layout) {
      console.error('[DEBUG zoom-in-btn] Chart element or layout not found');
      return;
    }

    try {
      // Get current range
      const currentXRange = chartEl.layout.xaxis?.range;
      const currentYRange = chartEl.layout.yaxis?.range;

      if (!currentXRange || !currentYRange || currentXRange.length !== 2 || currentYRange.length !== 2) {
        console.error('[DEBUG zoom-in-btn] Current range not available');
        return;
      }

      // Calculate center point
      const xStart = new Date(currentXRange[0]).getTime();
      const xEnd = new Date(currentXRange[1]).getTime();
      const xCenter = (xStart + xEnd) / 2;
      const xSpan = (xEnd - xStart) / 2;
      const newXSpan = xSpan / 2; // Zoom in by 50%

      const yStart = currentYRange[0];
      const yEnd = currentYRange[1];
      const yCenter = (yStart + yEnd) / 2;
      const ySpan = (yEnd - yStart) / 2;
      const newYSpan = ySpan / 2; // Zoom in by 50%

      // Calculate new ranges
      const newXStart = new Date(xCenter - newXSpan).toISOString();
      const newXEnd = new Date(xCenter + newXSpan).toISOString();
      const newYStart = yCenter - newYSpan;
      const newYEnd = yCenter + newYSpan;

      console.log('[DEBUG zoom-in-btn] Zooming in:', {
        oldXRange: currentXRange,
        newXRange: [newXStart, newXEnd],
        oldYRange: currentYRange,
        newYRange: [newYStart, newYEnd]
      });

      // Apply new range
      Plotly.relayout(chartEl, {
        'xaxis.range': [newXStart, newXEnd],
        'yaxis.range': [newYStart, newYEnd]
      });
    } catch (err) {
      console.error('[DEBUG zoom-in-btn] Error during zoom in:', err);
    }
  });

  document.getElementById('zoom-out-btn').addEventListener('click', () => {
    const chartEl = document.getElementById('chart');
    if (!chartEl || !chartEl.layout) {
      console.error('[DEBUG zoom-out-btn] Chart element or layout not found');
      return;
    }

    try {
      // Get current range
      const currentXRange = chartEl.layout.xaxis?.range;
      const currentYRange = chartEl.layout.yaxis?.range;

      if (!currentXRange || !currentYRange || currentXRange.length !== 2 || currentYRange.length !== 2) {
        console.error('[DEBUG zoom-out-btn] Current range not available');
        return;
      }

      // Calculate center point
      const xStart = new Date(currentXRange[0]).getTime();
      const xEnd = new Date(currentXRange[1]).getTime();
      const xCenter = (xStart + xEnd) / 2;
      const xSpan = (xEnd - xStart) / 2;
      const newXSpan = xSpan * 2; // Zoom out by 100%

      const yStart = currentYRange[0];
      const yEnd = currentYRange[1];
      const yCenter = (yStart + yEnd) / 2;
      const ySpan = (yEnd - yStart) / 2;
      const newYSpan = ySpan * 2; // Zoom out by 100%

      // Calculate new ranges
      const newXStart = new Date(xCenter - newXSpan).toISOString();
      const newXEnd = new Date(xCenter + newXSpan).toISOString();
      const newYStart = yCenter - newYSpan;
      const newYEnd = yCenter + newYSpan;

      console.log('[DEBUG zoom-out-btn] Zooming out:', {
        oldXRange: currentXRange,
        newXRange: [newXStart, newXEnd],
        oldYRange: currentYRange,
        newYRange: [newYStart, newYEnd]
      });

      // Apply new range
      Plotly.relayout(chartEl, {
        'xaxis.range': [newXStart, newXEnd],
        'yaxis.range': [newYStart, newYEnd]
      });
    } catch (err) {
      console.error('[DEBUG zoom-out-btn] Error during zoom out:', err);
    }
  });

  document.getElementById('draw-rect-btn').addEventListener('click', () => {
    Plotly.relayout('chart', {
      dragmode: 'drawrect'
    });
  });

  document.getElementById('draw-line-btn').addEventListener('click', () => {
    Plotly.relayout('chart', {
      dragmode: 'drawline'
    });
  });

  async function loadViewRange() {
    try {
      console.log('[DEBUG loadViewRange] Loading view range from server...');
      const resp = await fetch('/view-range');
      if (!resp.ok) {
        console.error('[DEBUG loadViewRange] Server returned error status:', resp.status, resp.statusText);
        return {};
      }
      const range = await resp.json();
      
      console.log('[DEBUG loadViewRange] Raw response from server:', range);
      console.log('[DEBUG loadViewRange] Range xaxis:', range.xaxis);
      console.log('[DEBUG loadViewRange] Range yaxis:', range.yaxis);
      
      // Validate the loaded range to prevent using corrupted data
      if (range && range.xaxis && Array.isArray(range.xaxis.range) && range.xaxis.range.length === 2) {
        const [startRaw, endRaw] = range.xaxis.range;
        
        console.log('[DEBUG loadViewRange] Loaded X-axis range:', {
          startRaw: startRaw,
          endRaw: endRaw,
          startDate: new Date(startRaw).toISOString(),
          endDate: new Date(endRaw).toISOString()
        });
        
        // Convert to timestamps and validate
        const startTime = new Date(startRaw).getTime();
        const endTime = new Date(endRaw).getTime();
        
        console.log('[DEBUG loadViewRange] Converted to timestamps:', {
          startTime: startTime,
          endTime: endTime
        });
        
        // Check if we have valid timestamps
        const isValid = Number.isFinite(startTime) && Number.isFinite(endTime) && 
                       startTime > 0 && endTime > 0 && startTime < endTime;
        
        if (!isValid) {
          console.warn('Invalid view range detected from server, clearing corrupted data:', startRaw, endRaw);
          console.warn('[DEBUG loadViewRange] Validation failed:', {
            startTimeValid: Number.isFinite(startTime),
            endTimeValid: Number.isFinite(endTime),
            startTimePositive: startTime > 0,
            endTimePositive: endTime > 0,
            startBeforeEnd: startTime < endTime
          });
          // Clear the corrupted data by saving a default range
          await saveDefaultViewRange();
          return {};
        }
        
        // Additional validation: ensure reasonable time range
        const timeRangeMs = endTime - startTime;
        const minRangeMs = 60 * 1000; // 1 minute minimum
        const maxRangeMs = 5 * 365 * 24 * 60 * 60 * 1000; // 5 year maximum
        
        if (timeRangeMs < minRangeMs || timeRangeMs > maxRangeMs) {
          console.warn('View range too small or too large, clearing corrupted data:', timeRangeMs);
          // Clear the corrupted data by saving a default range
          await saveDefaultViewRange();
          return {};
        }
      } else {
        console.log('[DEBUG loadViewRange] No valid range found, returning empty object');
      }
      
      return range;
    } catch (err) {
      console.error('Failed to load view range:', err);
      return {};
    }
  }
  
  async function saveDefaultViewRange() {
    try {
      // Save a default view range (last 10 days)
      const now = new Date();
      const tenDaysAgo = new Date(now.getTime() - 10 * 24 * 60 * 60 * 1000);
      
      const defaultRange = {
        xaxis: {
          range: [tenDaysAgo.toISOString(), now.toISOString()]
        },
        yaxis: {
          range: null
        }
      };
      
      await fetch('/view-range', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(defaultRange),
      });
      
      console.log('Saved default view range');
    } catch (err) {
      console.error('Failed to save default view range:', err);
    }
  }

  async function saveViewRange(range) {
    try {
      console.log('[DEBUG saveViewRange] Called with range:', range);
      console.log('[DEBUG saveViewRange] Range dates:', {
        xaxis: range.xaxis?.range ? range.xaxis.range.map(v => new Date(v).toISOString()) : 'null/undefined',
        yaxis: range.yaxis?.range ? range.yaxis.range.map(v => new Date(v).toISOString()) : 'null/undefined'
      });

      // Handle autoscale case where ranges are null
      if (range && (range.xaxis?.range === null || range.yaxis?.range === null)) {
        console.log('Saving autoscale view range (null ranges)');
        await fetch('/view-range', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(range),
        });
        return;
      }
      
      // Validate the range before saving to prevent corruption
      if (range && range.xaxis && Array.isArray(range.xaxis.range) && range.xaxis.range.length === 2) {
        const [startRaw, endRaw] = range.xaxis.range;
        
        console.log('[DEBUG saveViewRange] Validating range:', {
          startRaw: startRaw,
          endRaw: endRaw,
          startDate: new Date(startRaw).toISOString(),
          endDate: new Date(endRaw).toISOString()
        });
        
        // Convert to timestamps and validate
        const startTime = new Date(startRaw).getTime();
        const endTime = new Date(endRaw).getTime();
        
        console.log('[DEBUG saveViewRange] Converted timestamps:', {
          startTime: startTime,
          endTime: endTime
        });
        
        // Check if we have valid timestamps
        const isValid = Number.isFinite(startTime) && Number.isFinite(endTime) &&
                       startTime > 0 && endTime > 0 && startTime < endTime;
        
        if (!isValid) {
          console.warn('Invalid view range detected, not saving to server:', startRaw, endRaw);
          console.warn('[DEBUG saveViewRange] Validation details:', {
            startTimeValid: Number.isFinite(startTime),
            endTimeValid: Number.isFinite(endTime),
            startTimePositive: startTime > 0,
            endTimePositive: endTime > 0,
            startBeforeEnd: startTime < endTime
          });
          return;
        }
        
        // Additional validation: ensure reasonable time range
        const timeRangeMs = endTime - startTime;
        const minRangeMs = 60 * 1000; // 1 minute minimum
        const maxRangeMs = 5 * 365 * 24 * 60 * 60 * 1000; // 5 year maximum
        
        if (timeRangeMs < minRangeMs || timeRangeMs > maxRangeMs) {
          console.warn('View range too small or too large, not saving:', timeRangeMs);
          return;
        }
      } else {
        console.log('[DEBUG saveViewRange] No array range to validate');
      }
      
      console.log('[DEBUG saveViewRange] Saving range to server:', range);
      await fetch('/view-range', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(range),
      });
      console.log('[DEBUG saveViewRange] Range saved successfully');
    } catch (err) {
      console.error('Failed to save view range:', err);
    }
  }

  async function loadDrawings(symbol) {
    try {
      const resp = await fetch(`/drawings?symbol=${encodeURIComponent(symbol)}`);
      if (!resp.ok) {
        console.error('Failed to load drawings:', resp.status);
        return [];
      }
      const result = await resp.json();
      return result.shapes || [];
    } catch (err) {
      console.error('Failed to load drawings:', err);
      return [];
    }
  }

  async function saveDrawings(symbol) {
    try {
      const chartEl = document.getElementById('chart');
      if (!chartEl || !chartEl._fullLayout) {
        console.log('[saveDrawings] Chart not ready, aborting save');
        return;
      }
      const shapes = chartEl._fullLayout.shapes || [];
      const customShapes = shapes.filter(shape => {
        // Exclude live price line (marked with isLivePrice flag)
        if (shape.isLivePrice === true) {
          return false;
        }
        // Also exclude by pattern: x0=0, x1=1, red dashed line
        if (shape.type === 'line' && shape.x0 === 0 && shape.x1 === 1 && 
            shape.line?.color === '#FF6B6B' && shape.line?.dash === 'dash') {
          return false;
        }
        // Include all other lines and rectangles
        return (shape.type === 'line' || shape.type === 'rect' || shape.type === 'rectangle');
      });

      // Streamline shapes to only essential properties
      const streamlinedShapes = customShapes.map(shape => {
        const baseShape = {
          type: shape.type,
          x0: shape.x0,
          x1: shape.x1,
          y0: shape.y0,
          y1: shape.y1,
          line: {
            color: shape.line?.color || 'black',
            width: shape.line?.width || 2,
            dash: shape.line?.dash || 'solid'
          }
        };

        // Add sendEmailOnCross for lines
        if (shape.type === 'line') {
          baseShape.sendEmailOnCross = shape.sendEmailOnCross || false;
        }

        return baseShape;
      });

      console.log('[saveDrawings] Saving drawings - symbol:', symbol, 'total shapes:', shapes.length, 'custom shapes:', customShapes.length);
      console.log('[saveDrawings] Streamlined shapes to save:', JSON.stringify(streamlinedShapes, null, 2));

      const response = await fetch('/drawings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: symbol,
          shapes: streamlinedShapes
        }),
      });

      console.log('[saveDrawings] Server response status:', response.status, response.statusText);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('[saveDrawings] Server returned error:', errorText);
      } else {
        const result = await response.json();
        console.log('[saveDrawings] Server response:', result);
      }
    } catch (err) {
      console.error('[saveDrawings] Failed to save drawings:', err);
    }
  }

  const CLICK_THRESHOLD_PX = 25;
  const HOVER_THRESHOLD_PX = 18;

  function ensureShapeEmailPropertyDefaults(chartEl) {
    if (!chartEl || !chartEl.layout || !chartEl._fullLayout) {
      return;
    }
    const layoutShapes = chartEl.layout.shapes || [];
    const fullLayoutShapes =
      Array.isArray(chartEl._fullLayout.shapes) ? chartEl._fullLayout.shapes : [];
    const maxLen = Math.max(layoutShapes.length, fullLayoutShapes.length);

    for (let i = 0; i < maxLen; i++) {
      const layoutShape = layoutShapes[i];
      const fullShape = fullLayoutShapes[i];
      const s = layoutShape || fullShape;
      if (!s || s.type !== 'line') {
        continue;
      }
      if (typeof s.sendEmailOnCross === 'undefined') {
        const defaultValue = false;
        if (layoutShape) {
          layoutShape.sendEmailOnCross = defaultValue;
        }
        if (fullShape) {
          fullShape.sendEmailOnCross = defaultValue;
        }
      }
    }
  }

  function formatShapeCoord(value, isX) {
    if (value === undefined || value === null) {
      return 'N/A';
    }

    if (isX) {
      const d = new Date(value);
      if (!Number.isNaN(d.getTime())) {
        return d.toISOString();
      }
    }

    const num = Number(value);
    if (Number.isFinite(num)) {
      return num.toFixed(4);
    }
    return String(value);
  }

  function updateShapePropertiesPanel() {
    if (!shapeStatusEl || !shapeCoordinatesEl) {
      return;
    }

    const chartEl = document.getElementById('chart');
    if (!chartEl || !chartEl.layout) {
      shapeStatusEl.textContent = 'No shape selected.';
      shapeCoordinatesEl.innerHTML = '';
      if (shapeLineOptionsEl && shapeSendEmailCheckbox) {
        shapeLineOptionsEl.style.display = 'none';
        shapeSendEmailCheckbox.checked = false;
        shapeSendEmailCheckbox.disabled = true;
      }
      if (deleteShapeBtn) {
        deleteShapeBtn.disabled = true;
      }
      return;
    }

    const layoutShapes = chartEl.layout.shapes || [];
    const fullLayoutShapes =
      chartEl._fullLayout && Array.isArray(chartEl._fullLayout.shapes)
        ? chartEl._fullLayout.shapes
        : [];
    const shapes = layoutShapes.length ? layoutShapes : fullLayoutShapes;

    if (
      selectedShapeIndex === null ||
      !Array.isArray(shapes) ||
      !shapes[selectedShapeIndex]
    ) {
      shapeStatusEl.textContent = 'No shape selected.';
      shapeCoordinatesEl.innerHTML = '';
      if (shapeLineOptionsEl && shapeSendEmailCheckbox) {
        shapeLineOptionsEl.style.display = 'none';
        shapeSendEmailCheckbox.checked = false;
        shapeSendEmailCheckbox.disabled = true;
      }
      if (deleteShapeBtn) {
        deleteShapeBtn.disabled = true;
      }
      return;
    }

    const shape = shapes[selectedShapeIndex];
    const type = shape.type || 'shape';
    shapeStatusEl.textContent = `Selected shape: ${type}`;

    const x0 = formatShapeCoord(shape.x0, true);
    const y0 = formatShapeCoord(shape.y0, false);
    const x1 = formatShapeCoord(shape.x1, true);
    const y1 = formatShapeCoord(shape.y1, false);

    const label1 =
      type === 'rect' || type === 'rectangle' ? 'Corner 1' : 'Point 1';
    const label2 =
      type === 'rect' || type === 'rectangle' ? 'Corner 2' : 'Point 2';

    shapeCoordinatesEl.innerHTML =
      `<p>${label1}: (${x0}, ${y0})</p>` +
      `<p>${label2}: (${x1}, ${y1})</p>`;

    if (shapeLineOptionsEl && shapeSendEmailCheckbox) {
      if (type === 'line') {
        shapeLineOptionsEl.style.display = 'block';
        const currentValue =
          typeof shape.sendEmailOnCross === 'boolean'
            ? shape.sendEmailOnCross
            : false;
        shapeSendEmailCheckbox.disabled = false;
        shapeSendEmailCheckbox.checked = currentValue;
      } else {
        shapeLineOptionsEl.style.display = 'none';
        shapeSendEmailCheckbox.checked = false;
        shapeSendEmailCheckbox.disabled = true;
      }
    }
 
    if (deleteShapeBtn) {
      // We have a valid selected shape at this point, so enable deletion.
      deleteShapeBtn.disabled = false;
    }
  }

  // Utility: squared distance from point p to segment vw (in pixel coordinates)
  function distToSegmentSquared(p, v, w) {
    const vx = w.x - v.x;
    const vy = w.y - v.y;
    const l2 = vx * vx + vy * vy;
    if (l2 === 0) {
      const dx = p.x - v.x;
      const dy = p.y - v.y;
      return dx * dx + dy * dy;
    }
    let t = ((p.x - v.x) * vx + (p.y - v.y) * vy) / l2;
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    const projX = v.x + t * vx;
    const projY = v.y + t * vy;
    const dxp = p.x - projX;
    const dyp = p.y - projY;
    return dxp * dxp + dyp * dyp;
  }

  // Geometry-based hit-testing helper shared by click and hover.
  // Works in pixel coordinates using the full segment, not just the center.
  function findClosestShapeAtPoint(
    shapes,
    xaxis,
    yaxis,
    mouseX_paper,
    mouseY_paper,
    thresholdPx,
    logContext,
    logPerShapeDistances
  ) {
    const thresholdSq = thresholdPx * thresholdPx;
    let closestIndex = -1;
    let minDistSq = Infinity;

    const contextLabel = logContext === 'hover' ? '[shapes] hover' : '[shapes] click';

    if (
      !Array.isArray(shapes) ||
      !shapes.length ||
      !xaxis ||
      !yaxis ||
      typeof xaxis.d2p !== 'function' ||
      typeof yaxis.d2p !== 'function'
    ) {
      console.log(contextLabel, 'findClosestShapeAtPoint: no usable shapes or axes');
      return { closestIndex, minDistSq, thresholdSq, hitShape: false };
    }

    shapes.forEach((shape, i) => {
      if (!shape || (shape.type !== 'line' && shape.type !== 'rect')) {
        return;
      }

      let x0Val, x1Val, y0Val, y1Val;

      if (xaxis.type === 'date') {
        x0Val = shape.x0 instanceof Date ? shape.x0.getTime() : new Date(shape.x0).getTime();
        x1Val = shape.x1 instanceof Date ? shape.x1.getTime() : new Date(shape.x1).getTime();
      } else {
        x0Val = Number(shape.x0);
        x1Val = Number(shape.x1);
      }

      y0Val = Number(shape.y0);
      y1Val = Number(shape.y1);

      if (!Number.isFinite(x0Val) || !Number.isFinite(x1Val) || !Number.isFinite(y0Val) || !Number.isFinite(y1Val)) {
        console.log(contextLabel, 'findClosestShapeAtPoint: skipping shape with non-finite coords', i, shape);
        return;
      }

      if (shape.type === 'line') {
        const p0x = xaxis.d2p(x0Val);
        const p1x = xaxis.d2p(x1Val);
        const p0y = yaxis.d2p(y0Val);
        const p1y = yaxis.d2p(y1Val);

        const p0 = { x: xaxis._offset + p0x, y: yaxis._offset + p0y };
        const p1 = { x: xaxis._offset + p1x, y: yaxis._offset + p1y };

        if (!Number.isFinite(p0.x) || !Number.isFinite(p0.y) || !Number.isFinite(p1.x) || !Number.isFinite(p1.y)) {
          return;
        }

        const distSq = distToSegmentSquared({ x: mouseX_paper, y: mouseY_paper }, p0, p1);

        if (logPerShapeDistances) {
          console.log(contextLabel, 'line dist^2:', { index: i, distSq });
        }

        if (distSq < minDistSq) {
          minDistSq = distSq;
          closestIndex = i;
        }
      } else if (
        shape.type === 'rect' ||
        shape.type === 'rectangle' ||
        shape.type === 'box'
      ) {
        const x0p = xaxis.d2p(x0Val);
        const x1p = xaxis.d2p(x1Val);
        const y0p = yaxis.d2p(y0Val);
        const y1p = yaxis.d2p(y1Val);

        const left = xaxis._offset + Math.min(x0p, x1p);
        const right = xaxis._offset + Math.max(x0p, x1p);
        const top = yaxis._offset + Math.min(y0p, y1p);
        const bottom = yaxis._offset + Math.max(y0p, y1p);

        const inside =
          mouseX_paper >= left &&
          mouseX_paper <= right &&
          mouseY_paper >= top &&
          mouseY_paper <= bottom;

        if (inside) {
          // For rectangles we consider any click inside the bounds a "hit",
          // regardless of how far it is from the centre. To integrate with
          // the shared threshold-based logic we just use a distance of 0.
          const distSq = 0;

          if (logPerShapeDistances) {
            console.log(
              contextLabel,
              'rect hit (inside bounds), treating dist^2=0:',
              { index: i }
            );
          }

          if (distSq < minDistSq) {
            minDistSq = distSq;
            closestIndex = i;
          }
        }
      }
    });

    const hitShape = closestIndex !== -1 && minDistSq <= thresholdSq;
    // console.log(contextLabel, 'closestIndex:', closestIndex, 'minDistSq:', minDistSq, 'thresholdSq:', thresholdSq);

    return { closestIndex, minDistSq, thresholdSq, hitShape };
  }

  // Apply selection (red) and hover (green) styling in a single pass.
  function applyShapeStylesForSelectionAndHover(shapes, selectedIndex, hoveredIndex) {
    if (!Array.isArray(shapes) || !shapes.length) {
      return [];
    }

    return shapes.map((shape, i) => {
      const newShape = { ...shape };
      newShape.line = { ...(shape.line || {}) };

      // Remember the original color so we can restore it when deselecting / removing hover
      if (!newShape.line._originalColor) {
        const existingColor =
          (shape.line && shape.line._originalColor) ||
          (shape.line && shape.line.color) ||
          'black';

        // Treat plain 'red' as a previous "selected" state and normalise it back to black
        newShape.line._originalColor = existingColor === 'red' ? 'black' : existingColor;
      }

      // Remember the original width so we can restore it when deselecting / removing hover
      if (!newShape.line._originalWidth) {
        const existingWidth =
          (shape.line && shape.line._originalWidth) ||
          (shape.line && typeof shape.line.width === 'number' ? shape.line.width : 1);
        newShape.line._originalWidth = existingWidth;
      }

      const isSelected = selectedIndex !== null && i === selectedIndex;
      const isHovered = hoveredIndex !== null && i === hoveredIndex;

      if (isSelected) {
        // Selected shape: red and slightly thicker
        const baseWidth =
          typeof newShape.line._originalWidth === 'number'
            ? newShape.line._originalWidth
            : (typeof newShape.line.width === 'number' ? newShape.line.width : 1);
        newShape.line.color = 'red';
        newShape.line.width = Math.max(baseWidth + 1, 3);
      } else if (isHovered) {
        // Hovered shape: green, but keep original width
        newShape.line.color = 'green';
        if (typeof newShape.line._originalWidth === 'number') {
          newShape.line.width = newShape.line._originalWidth;
        }
      } else {
        // Neither selected nor hovered: original style
        newShape.line.color = newShape.line._originalColor;
        if (typeof newShape.line._originalWidth === 'number') {
          newShape.line.width = newShape.line._originalWidth;
        }
      }

      return newShape;
    });
  }

  // Click-based shape selection modeled after old project (chartInteractions.js: handleShapeClick)
  async function handleShapeClick(event) {
    const chartEl = document.getElementById('chart');
    const gd = chartEl;
    if (!gd || !gd._fullLayout || !gd.layout) {
      console.log('[shapes] handleShapeClick: missing gd or layout');
      return;
    }

    // Avoid interfering with drawing / dragging modes
    if (gd.layout.dragmode === 'drawline' || gd.layout.dragmode === 'drawrect') {
      return;
    }

    // Only react to primary-button clicks (ignore right/middle if present).
    // Note: Plotly sometimes dispatches synthetic click events with isTrusted=false;
    // we still want to use those as long as we can recover valid coordinates.
    if (event && typeof event.button === 'number' && event.button !== 0) {
      return;
    }

    // Plotly fires various synthetic click events; some have clientX/clientY = 0
    // but valid offsetX/offsetY. We want to *use* any coordinates we can, and
    // only bail out if we truly have nothing.
    let clientX = event.clientX;
    let clientY = event.clientY;

    const baseCoords = {
      clientX,
      clientY,
      offsetX: event.offsetX,
      offsetY: event.offsetY,
      pageX: event.pageX,
      pageY: event.pageY,
      type: event.type,
      isTrusted: event.isTrusted
    };
    console.log('[shapes] handleShapeClick raw event coords:', baseCoords);

    let hasValidClient =
      clientX != null &&
      clientY != null &&
      !(clientX === 0 && clientY === 0);

    // Fallback 1: use lastMouseEvent (real mousemove), like the old project.
    if (!hasValidClient &&
        window.lastMouseEvent &&
        window.lastMouseEvent.clientX != null &&
        window.lastMouseEvent.clientY != null &&
        !(window.lastMouseEvent.clientX === 0 && window.lastMouseEvent.clientY === 0)) {
      console.log(
        '[shapes] handleShapeClick using lastMouseEvent coords instead of synthetic click:',
        window.lastMouseEvent.clientX,
        window.lastMouseEvent.clientY
      );
      clientX = window.lastMouseEvent.clientX;
      clientY = window.lastMouseEvent.clientY;
      hasValidClient = true;
    }

    // Fallback 2: reconstruct from offsetX/offsetY relative to the event target.
    if (
      !hasValidClient &&
      typeof event.offsetX === 'number' &&
      typeof event.offsetY === 'number' &&
      event.target &&
      typeof event.target.getBoundingClientRect === 'function'
    ) {
      const targetRect = event.target.getBoundingClientRect();
      clientX = targetRect.left + event.offsetX;
      clientY = targetRect.top + event.offsetY;
      console.log(
        '[shapes] handleShapeClick using offset-based derived coords:',
        clientX,
        clientY,
        'offset:',
        event.offsetX,
        event.offsetY
      );
      hasValidClient = true;
    }

    if (!hasValidClient) {
      console.log('[shapes] handleShapeClick: no valid coordinates available (client, lastMouseEvent, or offset), aborting');
      return;
    }

    const rect = chartEl.getBoundingClientRect();
    const mouseX_div = clientX - rect.left;
    const mouseY_div = clientY - rect.top;

    console.log('[shapes] handleShapeClick fired, client:', clientX, clientY, 'local:', mouseX_div, mouseY_div);

    const mouseX_paper = mouseX_div;
    const mouseY_paper = mouseY_div;

    const fullLayout = gd._fullLayout;
    const xaxis = fullLayout.xaxis;
    const yaxis = fullLayout.yaxis;

    if (!xaxis || !yaxis || typeof xaxis.d2p !== 'function' || typeof yaxis.d2p !== 'function') {
      console.log('[shapes] handleShapeClick: missing xaxis/yaxis or d2p');
      return;
    }

    const shapesLayout = gd.layout.shapes || [];
    const shapesFull = fullLayout.shapes || [];
    const shapes = shapesLayout.length ? shapesLayout : shapesFull;

    if (!shapes.length) {
      console.log('[shapes] handleShapeClick: no shapes in layout or fullLayout');
      return;
    }

    // Ensure all line shapes have a sendEmailOnCross property
    ensureShapeEmailPropertyDefaults(gd);

    // Geometry-based hit testing (like the old project), but with a more
    // reasonable distance threshold so clicks near a line are treated as hits
    // without accidentally grabbing very distant shapes.
    const {
      closestIndex,
      minDistSq,
      thresholdSq,
      hitShape
    } = findClosestShapeAtPoint(
      shapes,
      xaxis,
      yaxis,
      mouseX_paper,
      mouseY_paper,
      CLICK_THRESHOLD_PX,
      'click',
      true
    );

    console.log('[shapes] handleShapeClick hit-test:', {
      mouseX_paper,
      mouseY_paper,
      closestIndex,
      minDistSq,
      thresholdSq,
      hitShape
    });

    // Decide what the new selected index should be (supports toggling & deselection)
    let newSelectedIndex = selectedShapeIndex;

    if (!hitShape) {
      // Clicked on empty chart area -> clear selection
      newSelectedIndex = null;
    } else if (selectedShapeIndex === closestIndex) {
      // Clicked again on the same shape -> toggle off
      newSelectedIndex = null;
    } else {
      // Selected a different shape
      newSelectedIndex = closestIndex;
    }

    const updatedShapes = applyShapeStylesForSelectionAndHover(
      shapes,
      newSelectedIndex,
      hoveredShapeIndex
    );

    selectedShapeIndex = newSelectedIndex;

    try {
      console.log('[shapes] handleShapeClick applying relayout with selected index:', closestIndex, 'newSelectedIndex:', newSelectedIndex);
      // Keep both layout.shapes and _fullLayout.shapes in sync with our updated array
      gd.layout.shapes = updatedShapes;
      if (Array.isArray(fullLayout.shapes) && fullLayout.shapes.length === updatedShapes.length) {
        fullLayout.shapes = updatedShapes;
      }

      updateShapePropertiesPanel();

      await Plotly.relayout(gd, { shapes: updatedShapes });
      await saveDrawings(currentSymbol);
    } catch (err) {
      console.error('[shapes] handleShapeClick failed to update selected shape color:', err);
    }
  }

  function setupShapeClickSelection() {
    const chartEl = document.getElementById('chart');
    console.log('[shapes] setupShapeClickSelection() called, chartEl:', chartEl);
    if (!chartEl || !chartEl._fullLayout) {
      console.log('[shapes] Missing chartEl or _fullLayout on setupShapeClickSelection, aborting');
      return;
    }

    if (chartEl.__shapeClickHandlerAttached) {
      console.log('[shapes] chartDiv shape click handler already attached');
      return;
    }
    chartEl.__shapeClickHandlerAttached = true;

    console.log('[shapes] Attaching chartDiv click handler for shape selection');
    chartEl.addEventListener('click', handleShapeClick, { capture: true });

    // Track lastMouseEvent (like old project) so clicks with 0,0 coords can use real mouse position,
    // and also drive hover-based highlighting.
    chartEl.addEventListener(
      'mousemove',
      (evt) => {
        window.lastMouseEvent = evt;
        handleShapeHover(evt);
      },
      { capture: true }
    );

    // Also track mouse movement at the document level so we still get real
    // coordinates even when Plotly's internal overlays consume the chart's
    // mousemove events. handleShapeClick() can then fall back to this when
    // click events have bogus (0,0) client coordinates.
    document.addEventListener(
      'mousemove',
      (evt) => {
        window.lastMouseEvent = evt;
      },
      { capture: true }
    );
  }

  // Hover-based shape highlighting (green), independent of selection.
  function handleShapeHover(event) {
    const chartEl = document.getElementById('chart');
    const gd = chartEl;
    if (!gd || !gd._fullLayout || !gd.layout) {
      return;
    }

    // Avoid interfering while user is actively drawing new shapes
    if (gd.layout.dragmode === 'drawline' || gd.layout.dragmode === 'drawrect') {
      return;
    }

    if (typeof event.clientX !== 'number' || typeof event.clientY !== 'number') {
      return;
    }

    const rect = chartEl.getBoundingClientRect();
    const mouseX_div = event.clientX - rect.left;
    const mouseY_div = event.clientY - rect.top;

    const mouseX_paper = mouseX_div;
    const mouseY_paper = mouseY_div;

    const fullLayout = gd._fullLayout;
    const xaxis = fullLayout.xaxis;
    const yaxis = fullLayout.yaxis;

    if (!xaxis || !yaxis || typeof xaxis.d2p !== 'function' || typeof yaxis.d2p !== 'function') {
      return;
    }

    const shapesLayout = gd.layout.shapes || [];
    const shapesFull = fullLayout.shapes || [];
    const shapes = shapesLayout.length ? shapesLayout : shapesFull;

    if (!shapes.length) {
      return;
    }

    const {
      closestIndex,
      minDistSq,
      thresholdSq,
      hitShape
    } = findClosestShapeAtPoint(
      shapes,
      xaxis,
      yaxis,
      mouseX_paper,
      mouseY_paper,
      HOVER_THRESHOLD_PX,
      'hover',
      false
    );

    const newHoveredIndex = hitShape ? closestIndex : null;

    // Only relayout when the hovered index actually changes
    if (newHoveredIndex === hoveredShapeIndex) {
      return;
    }

    hoveredShapeIndex = newHoveredIndex;

    console.log('[shapes] handleShapeHover:', {
      mouseX_paper,
      mouseY_paper,
      hoveredShapeIndex,
      minDistSq,
      thresholdSq,
      hitShape
    });

    const updatedShapes = applyShapeStylesForSelectionAndHover(
      shapes,
      selectedShapeIndex,
      hoveredShapeIndex
    );

    if (!updatedShapes || !updatedShapes.length) {
      return;
    }

    try {
      gd.layout.shapes = updatedShapes;
      if (Array.isArray(fullLayout.shapes) && fullLayout.shapes.length === updatedShapes.length) {
        fullLayout.shapes = updatedShapes;
      }
      // Visual-only: do not persist hover styling to backend
      Plotly.relayout(gd, { shapes: updatedShapes });
    } catch (err) {
      console.error('[shapes] handleShapeHover failed to update hover style:', err);
    }
  }

  function downsampleOhlcSeries(data, maxPoints) {
    const len = data.time.length;
    if (len <= maxPoints) {
      return data;
    }

    const bucketSize = Math.ceil(len / maxPoints);
    const result = {
      time: [],
      open: [],
      high: [],
      low: [],
      close: []
    };

    for (let start = 0; start < len; start += bucketSize) {
      const end = Math.min(start + bucketSize, len);

      const bucketOpenTime = data.time[start];
      const bucketOpen = data.open[start];
      let bucketHigh = data.high[start];
      let bucketLow = data.low[start];
      const bucketClose = data.close[end - 1];

      for (let i = start + 1; i < end; i++) {
        if (data.high[i] > bucketHigh) bucketHigh = data.high[i];
        if (data.low[i] < bucketLow) bucketLow = data.low[i];
      }

      result.time.push(bucketOpenTime);
      result.open.push(bucketOpen);
      result.high.push(bucketHigh);
      result.low.push(bucketLow);
      result.close.push(bucketClose);
    }

    return result;
  }

  /**
   * Create Plotly traces representing volume profile bars inside rectangles.
   *
   * rectVolumeProfiles: array of objects from /data or /volume_profile:
   *   {
   *     shape_index: Number,
   *     x0: ms,
   *     x1: ms,
   *     y0: Number,
   *     y1: Number,
   *     volume_profile: [{ price, totalVolume }]
   *   }
   *
   * shapes: Plotly layout shapes array for the current chart.
   */
  function createRectangleVolumeProfileBars(rectVolumeProfiles, shapes) {
    if (
      !Array.isArray(rectVolumeProfiles) ||
      !rectVolumeProfiles.length ||
      !Array.isArray(shapes) ||
      !shapes.length
    ) {
      return [];
    }

    const traces = [];

    rectVolumeProfiles.forEach((entry) => {
      if (!entry || !Array.isArray(entry.volume_profile) || !entry.volume_profile.length) {
        return;
      }

      const shapeIndex = typeof entry.shape_index === 'number' ? entry.shape_index : null;
      if (shapeIndex === null || shapeIndex < 0 || shapeIndex >= shapes.length) {
        return;
      }

      const shape = shapes[shapeIndex];
      if (!shape || (shape.type !== 'rect' && shape.type !== 'rectangle' && shape.type !== 'box')) {
        return;
      }

      const x0Ms = Number(entry.x0);
      const x1Ms = Number(entry.x1);
      const y0 = Number(entry.y0);
      const y1 = Number(entry.y1);

      if (!Number.isFinite(x0Ms) || !Number.isFinite(x1Ms) ||
          !Number.isFinite(y0) || !Number.isFinite(y1)) {
        return;
      }

      const rectTimeMin = Math.min(x0Ms, x1Ms);
      const rectTimeMax = Math.max(x0Ms, x1Ms);
      const priceMin = Math.min(y0, y1);
      const priceMax = Math.max(y0, y1);
      const priceRange = priceMax - priceMin;
      const timeRange = rectTimeMax - rectTimeMin;

      if (!Number.isFinite(priceRange) || priceRange <= 0 || !Number.isFinite(timeRange) || timeRange <= 0) {
        return;
      }

      // Normalise levels by price and volume
      const levels = entry.volume_profile
        .map((lvl) => ({
          price: Number(lvl.price),
          totalVolume: Number(lvl.totalVolume)
        }))
        .filter((lvl) =>
          Number.isFinite(lvl.price) &&
          lvl.price >= priceMin &&
          lvl.price <= priceMax &&
          Number.isFinite(lvl.totalVolume) &&
          lvl.totalVolume > 0
        )
        .sort((a, b) => a.price - b.price);

      if (!levels.length) {
        return;
      }

      const maxVolume = levels.reduce(
        (acc, lvl) => (lvl.totalVolume > acc ? lvl.totalVolume : acc),
        0
      );
      if (!(maxVolume > 0)) {
        return;
      }

      // Estimate bar thickness in price units based on gaps between levels.
      let minGap = null;
      for (let i = 0; i < levels.length - 1; i++) {
        const gap = levels[i + 1].price - levels[i].price;
        if (gap > 0 && (minGap === null || gap < minGap)) {
          minGap = gap;
        }
      }
      if (!Number.isFinite(minGap) || minGap <= 0) {
        minGap = priceRange / Math.max(levels.length, 1);
      }
      let barThickness = minGap * 0.8;
      const minThickness = priceRange * 0.001;
      const maxThickness = priceRange * 0.05;
      if (barThickness < minThickness) barThickness = minThickness;
      if (barThickness > maxThickness) barThickness = maxThickness;

      // Max horizontal extent for bars (as fraction of rectangle width)
      const maxBarTimeSpan = timeRange * 0.5; // 50% of rectangle width

      levels.forEach((lvl) => {
        const volumeRatio = lvl.totalVolume / maxVolume;
        const barTimeSpan = maxBarTimeSpan * volumeRatio;

        const leftTime = rectTimeMin;
        const rightTime = rectTimeMin + barTimeSpan;

        const centerPrice = lvl.price;
        let bottomPrice = centerPrice - barThickness / 2;
        let topPrice = centerPrice + barThickness / 2;

        if (bottomPrice < priceMin) bottomPrice = priceMin;
        if (topPrice > priceMax) topPrice = priceMax;

        const xCoords = [
          new Date(leftTime),
          new Date(rightTime),
          new Date(rightTime),
          new Date(leftTime)
        ];
        const yCoords = [
          bottomPrice,
          bottomPrice,
          topPrice,
          topPrice
        ];

        traces.push({
          x: xCoords,
          y: yCoords,
          type: 'scatter',
          mode: 'lines',
          line: { width: 0 },
          fill: 'toself',
          fillcolor: 'rgba(219, 175, 31, 0.4)',
          hoverinfo: 'text',
          name: `VP-rect-${shapeIndex} @ ${centerPrice.toFixed(4)}`,
          text: `Volume profile rectangle ${shapeIndex}
Price: ${centerPrice.toFixed(4)}
Volume: ${lvl.totalVolume}`,
          showlegend: false
        });
      });
    });

    return traces;
  }

  /**
   * Re-fetch rectangle volume profiles from the backend and redraw only
   * the volume-profile overlays, without reloading the main OHLC data.
   *
   * This is triggered after shape changes (draw / move / delete) so that
   * rectangle volume profiles stay in sync with the user's drawings.
   */
  async function refreshRectangleVolumeProfiles() {
    const chartEl = document.getElementById('chart');
    if (!chartEl || !chartEl.layout || !chartEl._fullLayout) {
      return;
    }

    const fullLayout = chartEl._fullLayout;
    const shapesLayout = chartEl.layout.shapes || [];
    const shapesFull = fullLayout.shapes || [];
    const shapes = shapesLayout.length ? shapesLayout : shapesFull;

    if (!Array.isArray(shapes) || !shapes.length) {
      return;
    }

    // Check if there is at least one rectangle-type shape.
    const hasRect = shapes.some((s) => {
      if (!s || typeof s.type !== 'string') return false;
      const t = s.type.toLowerCase();
      return t === 'rect' || t === 'rectangle' || t === 'box';
    });

    // If no rectangles remain, strip any existing volume-profile traces.
    if (!hasRect) {
      const existingData = chartEl.data || [];
      if (!existingData.length) {
        return;
      }
      const candleIndex = existingData.findIndex((t) => t && t.type === 'candlestick');
      if (candleIndex === -1) {
        return;
      }
      const candleTrace = existingData[candleIndex];
      const newData = [candleTrace];
      try {
        await Plotly.react(chartEl, newData, chartEl.layout);
      } catch (err) {
        console.error('[volume_profile] Failed to clear rectangle volume profiles:', err);
      }
      return;
    }

    // Derive current visible x-axis range (time window) if available.
    let xRange = null;
    if (
      chartEl.layout &&
      chartEl.layout.xaxis &&
      Array.isArray(chartEl.layout.xaxis.range) &&
      chartEl.layout.xaxis.range.length === 2 &&
      chartEl.layout.xaxis.range[0] != null &&
      chartEl.layout.xaxis.range[1] != null
    ) {
      xRange = chartEl.layout.xaxis.range;
    } else if (
      fullLayout.xaxis &&
      Array.isArray(fullLayout.xaxis.range) &&
      fullLayout.xaxis.range.length === 2 &&
      fullLayout.xaxis.range[0] != null &&
      fullLayout.xaxis.range[1] != null
    ) {
      xRange = fullLayout.xaxis.range;
    }

    let startTime;
    let endTime;
    if (Array.isArray(xRange) && xRange.length === 2) {
      const start = new Date(xRange[0]).getTime();
      const end = new Date(xRange[1]).getTime();
      if (Number.isFinite(start) && Number.isFinite(end) && start > 0 && end > 0) {
        startTime = start;
        endTime = end;
      } else {
        // Invalid timestamps detected, skip volume profile fetch
        console.log('[volume_profile] Invalid timestamps in range, skipping time-based volume profile fetch');
        return;
      }
    } else {
      // Handle autoscale case or no specific time range
      console.log('[volume_profile] Autoscale or no specific view range detected, skipping time-based volume profile fetch');
      return;
    }

    const params = new URLSearchParams();
    if (currentSymbol) {
      params.append('symbol', currentSymbol);
    }
    if (currentInterval) {
      params.append('interval', currentInterval);
    }

    if (Number.isFinite(startTime) && Number.isFinite(endTime)) {
      params.append('startTime', String(startTime));
      params.append('endTime', String(endTime));
    } else {
      // Fallback: use the current "limit" selection when no explicit time window.
      const limitSelectEl = document.getElementById('limit-select');
      const limitValue = limitSelectEl ? limitSelectEl.value : null;
      if (limitValue) {
        params.append('limit', String(limitValue));
      }
    }

    try {
      const url = `/volume_profile?${params.toString()}`;
      console.log('[volume_profile] Fetching updated rectangle volume profiles from', url);
      const resp = await fetch(url, { method: 'GET' });
      const result = await resp.json();

      if (!resp.ok) {
        console.error(
          '[volume_profile] Failed to fetch volume profile:',
          result && result.error ? result.error : resp.status
        );
        return;
      }

      const rectVolumeProfiles = Array.isArray(result.rect_volume_profiles)
        ? result.rect_volume_profiles
        : [];

      const vpTraces = createRectangleVolumeProfileBars(rectVolumeProfiles, shapes) || [];

      const existingData = chartEl.data || [];
      if (!existingData.length) {
        return;
      }

      // Keep the existing candlestick trace, replace any previous VP traces.
      const candleIndex = existingData.findIndex((t) => t && t.type === 'candlestick');
      const candleTrace = candleIndex !== -1 ? existingData[candleIndex] : existingData[0];
      const newData = [candleTrace, ...vpTraces];

      await Plotly.react(chartEl, newData, chartEl.layout);
    } catch (err) {
      console.error('[volume_profile] Error while refreshing rectangle volume profiles:', err);
    }
  }

  async function plotChart(data, range = {}, shapes = [], rectVolumeProfiles = []) {
    console.log('plotChart called with range:', range);
    // Removed debug logging that accessed range parameters

    // Removed range preservation logic to prevent infinite loop
    // Reset any in-memory shape selection whenever we fully (re)plot the chart
    selectedShapeIndex = null;
    hoveredShapeIndex = null;
    setProgress(0, 'Chart processing progress:');

    // Downsample data if necessary using OHLC aggregation
    const originalLength = data.time.length;
    let renderedData = data;
    if (originalLength > 10000) {
      const maxPoints = 1500;
      renderedData = downsampleOhlcSeries(data, maxPoints);
    }
    const renderedLength = renderedData.time.length;
    setProgress(25, 'Chart processing progress:');

    // Update data points display
    if (dataPointsSpan) {
      dataPointsSpan.textContent = `${originalLength} / ${renderedLength}`;
    }
    setProgress(50, 'Chart processing progress:');

    // Convert timestamps to ISO strings for consistent date handling
    const priceTrace = {
      x: renderedData.time.map(t => new Date(t).toISOString()),
      open: renderedData.open,
      high: renderedData.high,
      low: renderedData.low,
      close: renderedData.close,
      type: 'candlestick',
      name: currentSymbol
    };
    
    // Build xaxis configuration - NO programmatic range setting to prevent infinite loop
    const xaxisConfig = {
      title: 'Time',
      type: 'date',
      tickformat: '%Y-%m-%d %H:%M',
      hoverformat: '%Y-%m-%d %H:%M:%S',
      rangeslider: { visible: false }
    };
    
    // Build yaxis configuration - NO programmatic range setting to prevent infinite loop
    const yaxisConfig = {
      title: 'Price'
    };
    
    // Build layout with proper range handling
    const layout = {
      dragmode: 'pan',
      xaxis: xaxisConfig,
      yaxis: yaxisConfig,
      shapes: shapes || [],
      edits: {
        shapePosition: true
      },
      margin: {
        l: 40,
        r: 40,
        t: 10,
        b: 40
      },
      showlegend: false
    };

    // Removed all debug logging that accessed range properties
    const config = {
      responsive: true,
      displayModeBar: false,
      scrollZoom: true,
      editable: true,
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent'
    };

    // Combine main price trace with any rectangle volume profile traces
    let traces = [priceTrace];
    if (Array.isArray(rectVolumeProfiles) && rectVolumeProfiles.length && Array.isArray(shapes) && shapes.length) {
      const vpTraces = createRectangleVolumeProfileBars(rectVolumeProfiles, shapes);
      if (Array.isArray(vpTraces) && vpTraces.length) {
        traces = traces.concat(vpTraces);
      }
    }

    setProgress(75, 'Chart processing progress:');
    console.log('Calling Plotly.react');
    // Removed debug logging that accessed layout range properties

    Plotly.react('chart', traces, layout, config).then(() => {
      const chartEl = document.getElementById('chart');
      ensureShapeEmailPropertyDefaults(chartEl);
      updateXAxisRangeDisplay();
      
      // Initialize hash of user shapes to detect actual changes (not live price updates)
      if (chartEl && chartEl._fullLayout && Array.isArray(chartEl._fullLayout.shapes)) {
        const userShapes = chartEl._fullLayout.shapes.filter(s => {
          // Exclude live price line by flag
          if (s.isLivePrice === true) {
            return false;
          }
          // Also exclude by pattern: x0=0, x1=1, #FF6B6B, dash style
          if (s.type === 'line' && s.x0 === 0 && s.x1 === 1 && 
              s.line?.color === '#FF6B6B' && s.line?.dash === 'dash') {
            return false;
          }
          return true;
        });
        lastUserShapesHash = JSON.stringify(userShapes);
      }
      
      console.log('Chart plotted, adding relayout listener');
      // Add relayout listener
      document.getElementById('chart').on('plotly_relayout', async (eventdata) => {
        //console.log('plotly_relayout triggered:', eventdata);

        // Ignore relayout events during programmatic updates to prevent infinite loops
        if (window.isProgrammaticUpdate) {
          console.log('Ignoring relayout during programmatic update');
          return;
        }

        // Extract range values first to avoid initialization issues
        let xRange = eventdata['xaxis.range'] || (eventdata['xaxis.range[0]'] ? [eventdata['xaxis.range[0]'], eventdata['xaxis.range[1]']] : null);
        let yRange = eventdata['yaxis.range'] || (eventdata['yaxis.range[0]'] ? [eventdata['yaxis.range[0]'], eventdata['yaxis.range[1]']] : null);

        /*
        console.log('[RANGE_CHANGE] Relayout triggered with ranges:', {
          xRange: xRange,
          yRange: yRange,
          xRangeDates: Array.isArray(xRange) ? xRange.map(v => new Date(v).toISOString()) : xRange,
          yRangeValues: yRange
        });
        */

        // Check if this is a user-initiated zoom/pan operation
        // Only trigger for actual user interactions, not programmatic range changes
        const isUserZoomPan = eventdata &&
          (eventdata['xaxis.range[0]'] !== undefined || eventdata['yaxis.range[0]'] !== undefined ||
           eventdata['xaxis.range[1]'] !== undefined || eventdata['yaxis.range[1]'] !== undefined) &&
          // Add additional checks to distinguish user vs programmatic changes
          !eventdata['xaxis.autorange'] && !eventdata['yaxis.autorange'];

        if (isUserZoomPan) {
          console.log('User zoom/pan operation detected, fetching new data for zoomed range');
        } else {
          // Ignore programmatic range changes to prevent infinite loop
        }

        // Detect shape changes (draw/add/move/delete) and save drawings
        // BUT: ignore changes that only affect the live price line (isLivePrice flag)
        const shapeChangeKeys = eventdata
          ? Object.keys(eventdata).filter(key =>
              key === 'shapes' ||
              key.startsWith('shapes[') ||
              key.startsWith('shape.')
            )
          : [];
        
        let hasUserShapeChange = false;
        if (shapeChangeKeys.length > 0) {
          const chartEl = document.getElementById('chart');
          if (chartEl && chartEl._fullLayout && Array.isArray(chartEl._fullLayout.shapes)) {
            // Get only the user shapes (exclude live price line)
            const userShapes = chartEl._fullLayout.shapes.filter(s => {
              // Exclude if marked with isLivePrice flag
              if (s.isLivePrice === true) {
                return false;
              }
              // Also exclude by pattern: x0=0, x1=1, #FF6B6B, dash style
              if (s.type === 'line' && s.x0 === 0 && s.x1 === 1 && 
                  s.line?.color === '#FF6B6B' && s.line?.dash === 'dash') {
                return false;
              }
              return true;
            });
            
            // Create a hash of user shapes to detect actual changes
            const userShapesJson = JSON.stringify(userShapes);
            
            // Only consider it a user shape change if the user shapes have actually changed
            if (userShapesJson !== lastUserShapesHash) {
              hasUserShapeChange = true;
              lastUserShapesHash = userShapesJson;
            }
          }
        }

        if (hasUserShapeChange) {
          let totalShapes;
          const chartEl = document.getElementById('chart');
          if (chartEl && chartEl._fullLayout && Array.isArray(chartEl._fullLayout.shapes)) {
            totalShapes = chartEl._fullLayout.shapes.length;
          }

          if (chartEl) {
            ensureShapeEmailPropertyDefaults(chartEl);
          }

          if (selectedShapeIndex !== null) {
            updateShapePropertiesPanel();
          }

          console.log('[shapes] Shape change detected, saving drawings', {
            keys: shapeChangeKeys,
            totalShapes
          });
          try {
            await saveDrawings(currentSymbol);
          } catch (err) {
            console.error('Error saving drawings:', err);
          }

          // After shapes (including rectangles) change, re-fetch the latest
          // rectangle volume profiles from the backend and redraw only the
          // volume profile overlays. This uses the new /volume_profile
          // endpoint and does *not* reload the core OHLC data.
          try {
            await refreshRectangleVolumeProfiles();
          } catch (err) {
            console.error('Error refreshing rectangle volume profiles:', err);
          }
        }

        // DEBUG: Log extracted ranges with dates
        /*
        console.log('[DEBUG relayout] Extracted ranges from eventdata:', {
          xRange: xRange,
          yRange: yRange,
          xRangeDates: Array.isArray(xRange) ? xRange.map(v => new Date(v).toISOString()) : xRange,
          yRangeDates: Array.isArray(yRange) ? yRange.map(v => new Date(v).toISOString()) : yRange,
          eventdataKeys: Object.keys(eventdata)
        });
        */

        // When autoscale is triggered, Plotly emits *autorange* flags in eventdata
        // (e.g. "xaxis.autorange": true). For autoscale events, we should NOT
        // refetch data - just let Plotly auto-scale to the existing data and save
        // null ranges to indicate automatic scaling.
        if (!xRange && (eventdata['xaxis.autorange'] || eventdata['xaxis.autorange'] === true)) {
          console.log('Autoscale detected on X axis - letting Plotly auto-scale to existing data');
          xRange = null;
        }

        if (!yRange && (eventdata['yaxis.autorange'] || eventdata['yaxis.autorange'] === true)) {
          console.log('Autoscale detected on Y axis - letting Plotly auto-scale to existing data');
          yRange = null;
        }

        // Additional fix: if we have autoscale flags but xRange/yRange are still arrays
        // containing null values from the chart layout, set them to null explicitly
        if (Array.isArray(xRange) && xRange.length === 2 && (xRange[0] === null || xRange[1] === null)) {
          console.log('Detected null values in X axis range array, setting to null for autoscale');
          xRange = null;
        }

        if (Array.isArray(yRange) && yRange.length === 2 && (yRange[0] === null || yRange[1] === null)) {
          console.log('Detected null values in Y axis range array, setting to null for autoscale');
          yRange = null;
        }

        // For autoscale events, save null ranges but don't fetch new data
        // For regular range changes, save ranges and fetch new data
        const isAutoscaleEvent = (eventdata['xaxis.autorange'] || eventdata['xaxis.autorange'] === true || 
                                 eventdata['yaxis.autorange'] || eventdata['yaxis.autorange'] === true);
        
        // Update X-axis range display whenever it changes
        updateXAxisRangeDisplay();

        // Only debounce for user zoom/pan events, not programmatic range changes
        if (isUserZoomPan) {
          console.log('User zoom/pan detected, debouncing...');
          clearTimeout(debounceTimeout);
          debounceTimeout = setTimeout(async () => {
            if (currentAbortController) {
              const oldLoadId = activeLoadId;
              if (socket && activeLoadId) {
                socket.emit('cancel_load', { loadId: activeLoadId });
              }
              currentAbortController.abort();
              console.log(`[debounce] Aborting previous fetch for loadId: ${oldLoadId}`);
            }
            currentAbortController = new AbortController();

            // Always save the range (including null ranges for autoscale)
            await saveViewRange({ xaxis: { range: xRange }, yaxis: { range: yRange } });
            console.log('View range saved');

            // Only fetch new data for non-autoscale events with valid ranges
            if (!isAutoscaleEvent && xRange && Array.isArray(xRange) && xRange.length === 2) {
              console.log('Debounce timeout, fetching new data');
              const startTime = new Date(xRange[0]).getTime();
              const endTime = new Date(xRange[1]).getTime();
              console.log('Fetching data for range:', startTime, endTime);

              const loadId = ++currentLoadId;
              activeLoadId = loadId;
              lastDisplayedProgress = 0;
              setProgress(0, 'Loading data...');
              console.log(`[debounce] Starting fetch for loadId: ${loadId}, time range: ${startTime} to ${endTime}`);

              const url = `/data?symbol=${encodeURIComponent(currentSymbol)}&interval=${currentInterval}&startTime=${startTime}&endTime=${endTime}&loadId=${loadId}`;

              try {
                const resp = await fetch(url, { signal: currentAbortController.signal });
                const result = await resp.json();
                if (resp.ok) {
                  console.log(`[debounce] Fetch completed for loadId: ${loadId}, data length: ${result.time ? result.time.length : 'undefined'}`);
                  console.log('Data fetched, replotting');

                  // Preserve the current Y-range from the chart to prevent autoscaling
                  const chartEl = document.getElementById('chart');
                  const currentYRange = chartEl?.layout?.yaxis?.range;

                  // Calculate X-range from the actual data to prevent Plotly from adjusting it
                  const dataXValues = result.time.map(t => new Date(t).getTime()).filter(t => Number.isFinite(t));
                  const dataXMin = Math.min(...dataXValues);
                  const dataXMax = Math.max(...dataXValues);
                  const dataXRange = Number.isFinite(dataXMin) && Number.isFinite(dataXMax) && dataXMin < dataXMax
                    ? [new Date(dataXMin).toISOString(), new Date(dataXMax).toISOString()]
                    : xRange;

                  console.log('[RANGE_CHANGE] Replotting after data fetch:', {
                    requestedXRange: xRange,
                    dataXRange: dataXRange,
                    yRange: yRange,
                    currentYRange: currentYRange,
                    willUseYRange: currentYRange && Array.isArray(currentYRange) && currentYRange.length === 2 ? currentYRange : yRange
                  });

                  const drawings = await loadDrawings(currentSymbol);

                  // Set flag to indicate this is a programmatic update, not user action
                  window.isProgrammaticUpdate = true;

                  // Create new chart data with updated price data
                  const priceTrace = {
                    x: result.time.map(t => new Date(t).toISOString()),
                    open: result.open,
                    high: result.high,
                    low: result.low,
                    close: result.close,
                    type: 'candlestick',
                    name: currentSymbol
                  };

                  // Combine with volume profile traces if they exist
                  let traces = [priceTrace];
                  if (Array.isArray(result.rect_volume_profiles) && result.rect_volume_profiles.length && Array.isArray(drawings) && drawings.length) {
                    const vpTraces = createRectangleVolumeProfileBars(result.rect_volume_profiles, drawings);
                    if (Array.isArray(vpTraces) && vpTraces.length) {
                      traces = traces.concat(vpTraces);
                    }
                  }

                  // Create layout preserving the current ranges
                  const layout = {
                    dragmode: 'pan',
                    xaxis: {
                      title: 'Time',
                      type: 'date',
                      tickformat: '%Y-%m-%d %H:%M',
                      hoverformat: '%Y-%m-%d %H:%M:%S',
                      rangeslider: { visible: false },
                      range: xRange  // Preserve user's X-range
                    },
                    yaxis: {
                      title: 'Price',
                      range: currentYRange  // Preserve user's Y-range
                    },
                    shapes: drawings || [],
                    edits: {
                      shapePosition: true
                    },
                    margin: {
                      l: 40,
                      r: 40,
                      t: 10,
                      b: 40
                    },
                    showlegend: false
                  };

                  const config = {
                    responsive: true,
                    displayModeBar: false,
                    scrollZoom: true,
                    editable: true,
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent'
                  };

                  // Fully replace the chart with new data
                  await Plotly.react(chartEl, traces, layout, config);

                  // Clear flag after update
                  delete window.isProgrammaticUpdate;
                } else {
                  console.log(`[debounce] Fetch failed for loadId: ${loadId}, status: ${resp.status}`);
                  console.error('Failed to fetch data:', result.error);
                }
              } catch (err) {
                if (err.name === 'AbortError') {
                  console.log(`[debounce] Data fetch aborted for loadId: ${loadId} due to new pan`);
                  return;
                }
                console.log(`[debounce] Error fetching for loadId: ${loadId}:`, err);
                console.error('Error fetching data:', err);
              }
            } else {
              console.log('Autoscale detected - not fetching new data, letting Plotly auto-scale to existing data');
            }
          }, 1000);
        }
      });
      // Enable selection of drawn shapes (rectangles/lines) by click
      setupShapeClickSelection();
      setProgress(100, 'Chart processing progress:');
    });
  }


  async function loadData() {
    console.log('Loading data');
    try {
      const symbol = symbolSelect.value;
      const interval = intervalSelect.value;
      const limit = limitSelect.value;

      if (!symbol || !interval) return;

      const previousSymbol = currentSymbol;
      const symbolChanged = previousSymbol && previousSymbol !== symbol;

      currentSymbol = symbol;
      currentInterval = interval;
      currentSymbolSpan.textContent = symbol;
      currentIntervalSpan.textContent = interval;

      const loadId = ++currentLoadId;
      activeLoadId = loadId;
      lastDisplayedProgress = 0;
      setProgress(0, 'Loading data...');
      statusDisplay.textContent = 'Loading...';

      // Load view range first so the very first request is /view-range,
      // then use that range (if available) when requesting /data.
      const range = await loadViewRange();
      console.log('Loaded view range before data fetch:', range);
      console.log('[DEBUG loadData] Range being used for plotChart:', {
        xaxis: range.xaxis,
        yaxis: range.yaxis,
        xaxisDates: range.xaxis?.range ? range.xaxis.range.map(v => new Date(v).toISOString()) : 'null'
      });

      let url = `/data?symbol=${encodeURIComponent(symbol)}&interval=${interval}`;
      let isAutoscaleLoad = false;

      if (range && range.xaxis && Array.isArray(range.xaxis.range)) {
        const [startRaw, endRaw] = range.xaxis.range;
        const startTime = new Date(startRaw).getTime();
        const endTime = new Date(endRaw).getTime();

        console.log('[DEBUG loadData] Processing loaded range for data fetch:', {
          startRaw: startRaw,
          endRaw: endRaw,
          startDate: new Date(startRaw).toISOString(),
          endDate: new Date(endRaw).toISOString(),
          startTime: startTime,
          endTime: endTime
        });

        if (Number.isFinite(startTime) && Number.isFinite(endTime)) {
          url += `&startTime=${startTime}&endTime=${endTime}`;
        } else {
          console.log('Invalid time range from server, falling back to limit');
          url += `&limit=${limit}`;
        }
      } else if (range && range.xaxis && range.xaxis.range === null) {
        // Handle autoscale case - fetch more data for proper autoscaling
        console.log('Autoscale view range detected, fetching more data for autoscaling');
        isAutoscaleLoad = true;
        // Use a much larger limit for autoscale to ensure we have enough data points to autorange properly
        url += `&limit=${Math.max(parseInt(limit) * 10, 1000)}`;
      } else {
        url += `&limit=${limit}`;
      }

      // Attach the loadId to the request so the server can tag progress events
      url += `&loadId=${loadId}`;

      console.log('Fetching data from URL', url);
      const resp = await fetch(url);
      const result = await resp.json();
      console.log('Fetch response:', resp.ok, result);
      
      // Debug: Check the actual time data being returned
      if (result.time && Array.isArray(result.time) && result.time.length > 0) {
        console.log('[DEBUG fetchData] First 3 time values:', result.time.slice(0, 3).map(t => ({
          value: t,
          date: new Date(t).toISOString(),
          year: new Date(t).getFullYear()
        })));
        console.log('[DEBUG fetchData] Last 3 time values:', result.time.slice(-3).map(t => ({
          value: t,
          date: new Date(t).toISOString(),
          year: new Date(t).getFullYear()
        })));
        console.log('[DEBUG fetchData] Time range check:', {
          minTime: Math.min(...result.time),
          minDate: new Date(Math.min(...result.time)).toISOString(),
          maxTime: Math.max(...result.time), 
          maxDate: new Date(Math.max(...result.time)).toISOString()
        });
      } else {
        console.log('[DEBUG fetchData] No time data found in response');
      }
      
      if (!resp.ok) {
        throw new Error(result.error || 'Failed to fetch data');
      }
      dataPointsSpan.textContent = result.time.length;

      const drawings = await loadDrawings(symbol);
      console.log('Loaded drawings:', drawings);

      // Don't apply saved ranges when loading data - let Plotly autorange initially
      // Saved ranges should only be applied when user explicitly zooms/pans
      const initialRange = { xaxis: { autorange: true }, yaxis: { autorange: true } };

      await plotChart(
        result,
        initialRange,
        drawings,
        Array.isArray(result.rect_volume_profiles) ? result.rect_volume_profiles : []
      );

      // When the symbol changes, apply autoscale button functionality
      // This calculates explicit ranges from the data instead of using autorange
      if (symbolChanged) {
        try {
          console.log('[DEBUG loadData] Symbol changed, applying autoscale button functionality');
          const chartEl = document.getElementById('chart');
          if (!chartEl || !chartEl.data || !chartEl.data.length || !chartEl.data[0]) {
            console.error('[DEBUG loadData] Chart data not available for autoscale calculation');
          } else {
            const traceData = chartEl.data[0];
            if (!traceData.x || !traceData.x.length || !traceData.low || !traceData.high) {
              console.error('[DEBUG loadData] Trace missing x or price data');
            } else {
              // Calculate X-axis range (time) from data
              const xValues = traceData.x.map(x => new Date(x).getTime()).filter(t => Number.isFinite(t));
              const xMin = Math.min(...xValues);
              const xMax = Math.max(...xValues);

              if (!Number.isFinite(xMin) || !Number.isFinite(xMax) || xMin >= xMax) {
                console.error('[DEBUG loadData] Invalid X range calculated:', { xMin, xMax });
              } else {
                // Calculate Y-axis range (price) from OHLC data with padding
                const yValues = [...traceData.low, ...traceData.high].filter(y => Number.isFinite(y));
                const yMinBase = Math.min(...yValues);
                const yMaxBase = Math.max(...yValues);

                if (!Number.isFinite(yMinBase) || !Number.isFinite(yMaxBase) || yMinBase >= yMaxBase) {
                  console.error('[DEBUG loadData] Invalid Y range calculated:', { yMinBase, yMaxBase });
                } else {
                  // Add 5% padding to Y range for better visualization
                  const yPadding = (yMaxBase - yMinBase) * 0.05;
                  const yMin = Math.max(0, yMinBase - yPadding); // Don't go below 0 for prices
                  const yMax = yMaxBase + yPadding;

                  console.log('[DEBUG loadData] Calculated ranges after symbol change:', {
                    xMin: new Date(xMin).toISOString(),
                    xMax: new Date(xMax).toISOString(),
                    yMin,
                    yMax,
                    yPadding
                  });

                  // Set explicit ranges calculated from data (not autorange)
                  await Plotly.relayout(chartEl, {
                    'xaxis.range': [new Date(xMin).toISOString(), new Date(xMax).toISOString()],
                    'yaxis.range': [yMin, yMax]
                  });

                  console.log('[DEBUG loadData] Autoscale applied successfully after symbol change');
                }
              }
            }
          }
        } catch (err) {
          console.error('Error applying autoscale after symbol change:', err);
        }
      }

      statusDisplay.textContent = 'Loaded successfully';
    } catch (err) {
      console.error('Error in loadData:', err);
      statusDisplay.textContent = `Error: ${err.message}`;
      setProgress(0, `Error: ${err.message}`);
    }
  }

  // Add window resize listener to make chart responsive
  window.addEventListener('resize', () => {
    const chartEl = document.getElementById('chart');
    if (chartEl && chartEl.data) {
      Plotly.Plots.resize('chart');
    }
  });

  function updateLivePriceLine(price) {
    const chartEl = document.getElementById('chart');
    if (!chartEl || !chartEl.layout) {
      return;
    }

    // Remove existing live price annotations and shapes
    let shapes = (chartEl.layout.shapes || []).filter(s => !s.isLivePrice);
    let annotations = (chartEl.layout.annotations || []).filter(a => !a.isLivePrice);

    if (price > 0) {
      // Add horizontal line at current price
      shapes.push({
        type: 'line',
        x0: 0,
        x1: 1,
        y0: price,
        y1: price,
        xref: 'paper',
        yref: 'y',
        line: {
          color: '#FF6B6B',
          width: 2,
          dash: 'dash'
        },
        isLivePrice: true
      });

      // Add text annotation with price in a box
      annotations.push({
        x: 0.02,
        y: price,
        xref: 'paper',
        yref: 'y',
        text: `Live: $${price.toFixed(2)}`,
        showarrow: false,
        bgcolor: '#FF6B6B',
        bordercolor: '#CC0000',
        borderwidth: 2,
        borderpad: 8,
        font: {
          color: 'white',
          size: 12,
          family: 'Arial, sans-serif'
        },
        align: 'left',
        xanchor: 'left',
        yanchor: 'middle',
        isLivePrice: true
      });
    }

    // Update layout with new shapes and annotations
    Plotly.relayout(chartEl, {
      shapes: shapes,
      annotations: annotations
    }).catch(err => {
      console.error('[live_price] Error updating chart:', err);
    });
  }

  // Utility function to analyze timestamps for duplicates and patterns (backend mirror)
  function analyzeTimestamps(timestamps, symbol, interval) {
    const analysis = {
      total: timestamps.length,
      unique: new Set(timestamps).size,
      duplicates: [],
      patterns: [],
      hasDuplicates: false,
      duplicateCount: 0
    };

    // Find duplicates
    const seen = new Map();
    timestamps.forEach((ts, index) => {
      if (seen.has(ts)) {
        analysis.hasDuplicates = true;
        analysis.duplicateCount++;
        if (!analysis.duplicates.includes(ts)) {
          analysis.duplicates.push(ts);
        }
        // Track consecutive duplicates
        if (index > 0 && timestamps[index - 1] === ts) {
          analysis.patterns.push({
            timestamp: ts,
            consecutiveCount: (analysis.patterns.find(p => p.timestamp === ts)?.consecutiveCount || 1) + 1,
            date: new Date(ts).toISOString()
          });
        }
      } else {
        seen.set(ts, 1);
      }
    });

    // Remove duplicates from patterns, keeping only the longest streak
    const patternMap = new Map();
    analysis.patterns.forEach(p => {
      if (!patternMap.has(p.timestamp) || p.consecutiveCount > patternMap.get(p.timestamp).consecutiveCount) {
        patternMap.set(p.timestamp, p);
      }
    });
    analysis.patterns = Array.from(patternMap.values());

    console.log(`[TIMESTAMP_ANALYSIS] ${symbol}:${interval} - ${analysis.total} total, ${analysis.unique} unique, ${analysis.duplicateCount} duplicates`);
    if (analysis.duplicates.length > 0) {
      console.error('[TIMESTAMP_ANALYSIS] Duplicate timestamps:', analysis.duplicates.map(ts => ({
        timestamp: ts,
        date: new Date(ts).toISOString(),
        count: timestamps.filter(t => t === ts).length
      })));
    }
    if (analysis.patterns.length > 0) {
      console.error('[TIMESTAMP_ANALYSIS] Consecutive duplicates:', analysis.patterns);
    }

    return analysis;
  }

  // AI Features
  let aiSuggestionAbortController = null;

  async function fetchAndPopulateLocalOllamaModels() {
    try {
      const response = await fetch('/AI_Local_OLLAMA_Models');
      if (!response.ok) throw new Error(`Failed to fetch local models: ${response.status}`);
      const data = await response.json();
      localOllamaModelSelect.innerHTML = '';
      if (data.models && data.models.length > 0) {
        data.models.forEach(modelName => {
          const option = document.createElement('option');
          option.value = modelName;
          option.textContent = modelName;
          localOllamaModelSelect.appendChild(option);
        });
      } else {
        localOllamaModelSelect.innerHTML = '<option value="">No models found</option>';
      }
    } catch (error) {
      console.error("Error fetching local Ollama models:", error);
      localOllamaModelSelect.innerHTML = '<option value="">Error loading models</option>';
    }
  }

  function initializeAIFeatures() {
    console.log('Initializing AI features');
    if (!aiSuggestionBtn || !aiSuggestionTextarea || !useLocalOllamaCheckbox || !localOllamaModelDiv || !localOllamaModelSelect) {
      console.warn('AI elements not found, skipping AI initialization');
      return;
    }
    console.log('AI elements found, setting up event listeners');

    aiSuggestionBtn.addEventListener('click', async () => {
      console.log('AI suggestion button clicked');
      const chartEl = document.getElementById('chart');
      if (!chartEl || !chartEl.layout || !chartEl.layout.xaxis) {
        console.log('Chart not ready:', chartEl, chartEl?.layout, chartEl?.layout?.xaxis);
        alert("Chart not loaded yet or x-axis not defined.");
        aiSuggestionTextarea.value = "Error: Chart not ready.";
        return;
      }

      if (aiSuggestionBtn.textContent.startsWith("STOP")) {
        if (aiSuggestionAbortController) {
          aiSuggestionAbortController.abort();
        } else {
          aiSuggestionBtn.textContent = "Get AI Suggestion";
          aiSuggestionTextarea.value += "\n\n--- AI suggestion stop requested (no active process) ---";
        }
        return;
      }

      console.log('Starting AI suggestion request');
      aiSuggestionTextarea.value = "Getting AI suggestion, please wait...";
      aiSuggestionBtn.textContent = "STOP - Get AI Suggestion";

      try {
        const currentSymbol = symbolSelect.value;
        const currentResolution = intervalSelect.value;
        const plotlyXRange = chartEl.layout.xaxis.range;
        let xAxisMin, xAxisMax;

        if (plotlyXRange && plotlyXRange.length === 2) {
          const convertToTimestamp = (value) => {
            if (value instanceof Date) {
              if (isNaN(value.getTime())) return null;
              return Math.floor(value.getTime() / 1000);
            } else if (typeof value === 'string') {
              const parsedDate = new Date(value);
              if (isNaN(parsedDate.getTime())) return null;
              return Math.floor(parsedDate.getTime() / 1000);
            } else if (typeof value === 'number') {
              if (isNaN(value)) return null;
              return Math.floor(value / 1000);
            }
            return null;
          };
          xAxisMin = convertToTimestamp(plotlyXRange[0]);
          xAxisMax = convertToTimestamp(plotlyXRange[1]);
          if (xAxisMin === null || xAxisMax === null || isNaN(xAxisMin) || isNaN(xAxisMax)) {
            throw new Error("Could not parse valid time range from chart's x-axis.");
          }
        } else {
          throw new Error("Could not determine time range from chart's x-axis.");
        }

        // For now, no active indicators
        const activeIndicatorIds = [];

        const requestPayload = {
          symbol: currentSymbol,
          resolution: currentResolution,
          xAxisMin: xAxisMin,
          xAxisMax: xAxisMax,
          activeIndicatorIds: activeIndicatorIds,
          question: "Based on the provided market data, what is your trading suggestion (BUY, SELL, or HOLD) and why?",
          use_local_ollama: useLocalOllamaCheckbox.checked,
          local_ollama_model_name: localOllamaModelSelect.value || null
        };

        console.log('Sending request to /AI with payload:', requestPayload);
        aiSuggestionAbortController = new AbortController();
        const response = await fetch('/AI', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestPayload),
          signal: aiSuggestionAbortController.signal
        });
        console.log('Response status:', response.status);

        if (!response.ok) {
          const errorText = await response.text();
          console.log('Error response:', errorText);
          throw new Error(`AI suggestion failed: ${response.status} ${errorText}`);
        }

        if (useLocalOllamaCheckbox.checked) {
          aiChunks = {};
          finalChunks = {};
          aiSuggestionTextarea.value = "Streaming AI response...\n";
          aiComplete = false;
        } else {
          const result = await response.json();
          aiSuggestionTextarea.value = result.response || JSON.stringify(result, null, 2);
        }
      } catch (error) {
        console.log('AI suggestion error:', error);
        if (error.name === 'AbortError') {
          aiSuggestionTextarea.value += "\n\n--- AI suggestion stopped by user ---";
        } else {
          aiSuggestionTextarea.value = `Error: ${error.message}`;
        }
      } finally {
        aiSuggestionBtn.textContent = "Get AI Suggestion";
        aiSuggestionAbortController = null;
      }
    });

    useLocalOllamaCheckbox.addEventListener('change', function() {
      localOllamaModelDiv.style.display = this.checked ? 'block' : 'none';
      if (this.checked) fetchAndPopulateLocalOllamaModels();
    });
  }

  // Initialize AI features
  initializeAIFeatures();
});
