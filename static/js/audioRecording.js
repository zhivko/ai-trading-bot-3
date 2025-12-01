// Audio recording and transcription functionality

class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.stream = null;
        this.recordingStartTime = null;
        this.maxRecordingTime = 30000; // 30 seconds max
        this.recordingTimer = null;
    }

    async initialize() {
        try {
            // Request microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            return true;
        } catch (error) {
            console.error('❌ Error accessing microphone:', error);
            alert('Error accessing microphone. Please ensure you have granted microphone permissions.');
            return false;
        }
    }

    startRecording() {
        if (this.isRecording) {
            console.warn('⚠️ Already recording');
            return false;
        }

        try {
            // Clear previous recording
            this.audioChunks = [];

            // Create MediaRecorder
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            // Set up event handlers
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                this.handleRecordingStop();
            };

            // Start recording
            this.mediaRecorder.start(1000); // Collect data every second
            this.isRecording = true;
            this.recordingStartTime = Date.now();

            // Set up auto-stop timer
            this.recordingTimer = setTimeout(() => {
                if (this.isRecording) {
                    this.stopRecording();
                }
            }, this.maxRecordingTime);

            return true;
        } catch (error) {
            console.error('❌ Error starting recording:', error);
            alert('Error starting recording: ' + error.message);
            return false;
        }
    }

    stopRecording() {
        if (!this.isRecording) {
            console.warn('⚠️ Not currently recording');
            return false;
        }

        try {
            this.mediaRecorder.stop();
            this.isRecording = false;

            // Clear auto-stop timer
            if (this.recordingTimer) {
                clearTimeout(this.recordingTimer);
                this.recordingTimer = null;
            }

            return true;
        } catch (error) {
            console.error('❌ Error stopping recording:', error);
            return false;
        }
    }

    async handleRecordingStop() {
        try {
            // Create audio blob
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });

            // Create FormData for upload
            const formData = new FormData();
            formData.append('audio_file', audioBlob, 'recording.webm');

            // Add current chart state parameters
            formData.append('symbol', window.currentSymbolForStream || 'BTCUSDT');
            formData.append('resolution', window.currentResolution || '1h');

            // Add selected language
            const languageSelect = document.getElementById('language-select');
            if (languageSelect) {
                formData.append('language', languageSelect.value);
                console.log('Audio: Sending language:', languageSelect.value);
            }

            // Add xAxis range if available - send as ISO format timestamps
            if (window.currentXAxisRange && window.currentXAxisRange.length >= 2) {
                const xAxisMinISO = new Date(window.currentXAxisRange[0] * 1000).toISOString();
                const xAxisMaxISO = new Date(window.currentXAxisRange[1] * 1000).toISOString();
                formData.append('xAxisMin', xAxisMinISO);
                formData.append('xAxisMax', xAxisMaxISO);
                console.log('Audio: Sending xAxis range in ISO format:', xAxisMinISO, xAxisMaxISO);
            }

            // Add active indicators - extract just the IDs from the state objects
            const activeIndicatorIds = (window.active_indicatorsState || []).map(indicator => indicator.id);
            formData.append('activeIndicatorIds', JSON.stringify(activeIndicatorIds));

            // Add AI settings from config
            const useLocalOllama = window.useLocalOllama !== undefined ? window.useLocalOllama : false;
            const useGemini = window.useGemini !== undefined ? window.useGemini : false;
            formData.append('use_local_ollama', useLocalOllama);
            formData.append('use_gemini', useGemini);

            // Show processing status
            this.updateUIStatus('processing', 'Transcribing audio and analyzing...');

            // Clear previous transcription result
            const resultElement = document.getElementById('transcription-result');
            if (resultElement) {
                resultElement.innerHTML = '';
                resultElement.style.display = 'none';
            }

            // Send to server
            const response = await fetch('/transcribe_audio', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            }

            const result = await response.json();
            this.updateUIStatus('success', result.transcribed_text);
            console.log('Transcription result:', result);

        } catch (error) {
            console.error('❌ Error processing recording:', error);
            this.updateUIStatus('error', 'Error: ' + error.message);
        }
    }

    updateUIStatus(status, message) {
        const statusElement = document.getElementById('audio-status');
        const recordButton = document.getElementById('record-button');
        const transcriptionResult = document.getElementById('transcription-result');

        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `audio-status ${status}`;
        }

        if (recordButton) {
            if (status === 'recording') {
                recordButton.textContent = '⏹️ Stop Recording';
                recordButton.className = 'record-button recording';
            } else {
                recordButton.textContent = '🎤 Start Recording';
                recordButton.className = 'record-button';
            }
        }

        if (transcriptionResult && status !== 'recording') {
            transcriptionResult.style.display = 'block';
        }
    }

    displayTranscriptionResult(result) {
        const resultElement = document.getElementById('transcription-result');
        if (!resultElement) return;

        const transcriptionText = result.transcribed_text || 'No speech detected';
        const language = result.language || 'unknown';
        const confidence = result.confidence ? (result.confidence * 100).toFixed(1) : 'N/A';
        const llmAnalysis = result.llm_analysis || 'No analysis available';

        resultElement.innerHTML = `
            <div class="transcription-header">
                <strong>Transcription Result:</strong>
            </div>
            <div class="transcription-text">
                "${transcriptionText}"
            </div>
            <div class="transcription-meta">
                Language: ${language} | Confidence: ${confidence}%
            </div>
            <div class="transcription-header" style="margin-top: 15px;">
                <strong>AI Analysis:</strong>
            </div>
            <div class="transcription-analysis">
                ${llmAnalysis}
            </div>
        `;

        resultElement.style.display = 'block';

        // Keep visible - user can refresh page or click record again to hide
        // Auto-hide disabled so user can read LLM analysis
    }

    cleanup() {
        if (this.isRecording) {
            this.stopRecording();
        }

        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }

        if (this.recordingTimer) {
            clearTimeout(this.recordingTimer);
            this.recordingTimer = null;
        }
    }
}

// Global audio recorder instance
let audioRecorder = null;

    // Initialize audio recording functionality
    function initializeAudioRecording() {
// Create audio recorder instance
    audioRecorder = new AudioRecorder();

    // Set up record button
    const recordButton = document.getElementById('record-button');
    if (recordButton) {
        recordButton.addEventListener('click', async () => {
            if (!audioRecorder) {
                console.error('❌ Audio recorder not initialized');
                return;
            }

            if (audioRecorder.isRecording) {
                // Stop recording
                audioRecorder.stopRecording();
                audioRecorder.updateUIStatus('stopped', 'Recording stopped. Processing...');
            } else {
                // Start recording
                const initialized = await audioRecorder.initialize();
                if (initialized) {
                    const started = audioRecorder.startRecording();
                    if (started) {
                        audioRecorder.updateUIStatus('recording', 'Recording... Click to stop');
                    }
                }
            }
        });
    }

    // Set up cleanup on page unload
    window.addEventListener('beforeunload', () => {
        if (audioRecorder) {
            audioRecorder.cleanup();
        }
    });
}

// Check if browser supports required APIs
function checkAudioSupport() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('⚠️ Browser does not support getUserMedia API');
        return false;
    }

    if (!MediaRecorder) {
        console.warn('⚠️ Browser does not support MediaRecorder API');
        return false;
    }

    if (!FormData) {
        console.warn('⚠️ Browser does not support FormData API');
        return false;
    }

    return true;
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (checkAudioSupport()) {
        initializeAudioRecording();
    } else {
        console.error('❌ Audio recording not supported in this browser');
        const recordButton = document.getElementById('record-button');
        if (recordButton) {
            recordButton.disabled = true;
            recordButton.textContent = 'Audio Recording Not Supported';
        }
    }
});

// Export for global access
window.AudioRecorder = AudioRecorder;
window.audioRecorder = audioRecorder;