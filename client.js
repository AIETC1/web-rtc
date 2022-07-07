// peer connection
let pc = null;

function createPeerConnection() {
    pc = new RTCPeerConnection({sdpSemantics: 'unified-plan'});

    // connect video
    pc.addEventListener('track', function(evt) {
        console.log(evt)
        if (evt.track.kind == 'video') {
            console.log('Attach track to element video');
            document.getElementById('video').srcObject = evt.streams[0];
        }
        else {
            console.log('Attach track to element audio');
            document.getElementById('audio').srcObject = evt.streams[0];
        }
    });

    return pc;
}

function negotiate() {
    return pc.createOffer().then(function(offer) {
        console.log("A")
        return pc.setLocalDescription(offer);
    }).then(function() {
        console.log("B")
        // wait for ICE gathering to complete
        return new Promise(function(resolve) {
            if (pc.iceGatheringState === 'complete') {
                console.log("C")
                resolve();
            } else {
                console.log("D")
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        console.log("E")
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function() {
        console.log("F")
        let offer = pc.localDescription;

        return fetch('http://127.0.0.1:8080/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                video: document.getElementById('video-transform').value,
                audio: document.getElementById('audio-effect').value
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function(response) {
        console.log("G")
        return response.json();
    }).then(function(answer) {
        console.log("H", answer)
        return pc.setRemoteDescription(answer);
    }).catch(function(e) {
        console.log("I")
        alert(e);
    });
}

function start() {
    pc = createPeerConnection();

    let constraints = {audio: true, video: true,};

    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        stream.getTracks().forEach(function(track) {
            pc.addTrack(track, stream);
        });
        return negotiate();

    }, function(err) {
        alert('Could not acquire media: ' + err);
    });
}

function stop() {
    // close transceivers
    if (pc.getTransceivers) {
        pc.getTransceivers().forEach(function(transceiver) {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });
    }

    // close local video
    pc.getSenders().forEach(function(sender) {
        sender.track.stop();
    });

    // close peer connection
    setTimeout(function() {
        pc.close();
    }, 500);
}