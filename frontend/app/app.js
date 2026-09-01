// Standalone Mode: Direct access allowed for isolated dashboard testing
// (Strict token check active only when hosted alongside /3D root)
if (window.location.search.includes('mission=started')) {
    window.history.replaceState({}, document.title, window.location.pathname);
}



/**
 * SatQuery AI — Leaflet Satellite Engine & Center-Framed Bracket UI Controller
 * Vignan University (Deshmukhi / Hyderabad: [17.3425484, 78.716864])
 */
(function() {
    var VIGNAN_CAMPUS = [17.3425484, 78.716864];
    var map = null;

    // Fade out cloud fog overlay on load
    window.addEventListener('load', function() {
        var fogOut = document.getElementById('fog-transition-out');
        if (fogOut) fogOut.style.opacity = '0';
    });

    function initMap() {
        // Initialize map with regional overview
        map = L.map('map', {
            center: VIGNAN_CAMPUS,
            zoom: 10,
            zoomControl: false,
            attributionControl: true
        });

        // Pure Esri World Imagery (No street labels)
        L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
            maxZoom: 21,
            attribution: 'Google Maps Satellite'
        }).addTo(map);

        // Tactical Target Reticle
        var customIcon = L.divIcon({
            className: 'satquery-marker',
            html: '<div style="width:36px; height:36px; border:1px dashed #fff; border-radius:50%; box-shadow:0 0 14px rgba(255,255,255,0.6); animation:spin 12s linear infinite;"></div><div style="margin-top:4px; font-family:monospace; font-size:9px; font-weight:bold; color:#fff; background:rgba(0,0,0,0.8); padding:2px 7px; border:1px solid rgba(255,255,255,0.3); border-radius:3px; white-space:nowrap;">VIGNAN UNIVERSITY [17.3425° N, 78.7169° E]</div>',
            iconSize: [160, 60],
            iconAnchor: [80, 18]
        });
        L.marker(VIGNAN_CAMPUS, { icon: customIcon }).addTo(map);

        // Campus boundary polygon overlay
        var campusBounds = [
            [17.3450, 78.7135],
            [17.3460, 78.7200],
            [17.3395, 78.7215],
            [17.3385, 78.7145]
        ];
        L.polygon(campusBounds, {
            color: '#ffffff',
            weight: 1.5,
            dashArray: '4, 4',
            fillColor: '#ffffff',
            fillOpacity: 0.08
        }).addTo(map);

        // Fast cinematic Google Earth Zoom onto campus buildings
        setTimeout(function() {
            map.invalidateSize();
            map.flyTo(VIGNAN_CAMPUS, 17, {
                duration: 2.8,
                easeLinearity: 0.35
            });
        }, 150);

        // Slide in the center-framed bracket dashboard quickly
        setTimeout(function() {
            var hudRoot = document.getElementById('satquery-hud-root');
            if (hudRoot) hudRoot.classList.add('open');
        }, 1400);
    }

    function initUI() {
        var toggleBtn = document.getElementById('hud-toggle');
        var hudRoot = document.getElementById('satquery-hud-root');
        var closeBtn = document.getElementById('sidebar-close-btn');

        if (toggleBtn && hudRoot) {
            toggleBtn.addEventListener('click', function() {
                hudRoot.classList.toggle('open');
            });
        }
        if (closeBtn && hudRoot) {
            closeBtn.addEventListener('click', function() {
                hudRoot.classList.remove('open');
            });
        }

        // Mode toggles
        var modeBtns = document.querySelectorAll('.mode-btn');
        modeBtns.forEach(function(btn) {
            btn.addEventListener('click', function() {
                modeBtns.forEach(function(b) { b.classList.remove('active'); });
                btn.classList.add('active');
                logTerminal('Sensor mode switched to: ' + btn.dataset.mode.toUpperCase());
            });
        });

        // Submit query button
        var submitBtn = document.getElementById('submit-btn');
        var queryInput = document.getElementById('query-input');
        if (submitBtn && queryInput) {
            submitBtn.addEventListener('click', function() {
                var q = queryInput.value.trim() || 'Analyze vegetation index and flood boundary';
                logTerminal('SUBMIT: ' + q);
                logTerminal('POST /api/v1/query/orchestrate -> LangGraph Router');
                animatePipeline();
            });
        }
    }

    function logTerminal(msg) {
        var term = document.getElementById('terminal');
        if (!term) return;
        var now = new Date().toTimeString().split(' ')[0];
        var d = document.createElement('div');
        d.innerHTML = '<span style="color:rgba(255,255,255,0.4);">[' + now + ']</span> <span style="color:#fff;">ENGINE</span> ' + msg;
        term.appendChild(d);
        term.scrollTop = term.scrollHeight;
    }

    function animatePipeline() {
        var rows = document.querySelectorAll('.stage-row');
        rows.forEach(function(r, idx) {
            var dot = r.querySelector('.dot');
            dot.className = 'dot';
            setTimeout(function() {
                dot.className = 'dot active';
                if (idx > 0) {
                    rows[idx - 1].querySelector('.dot').className = 'dot done';
                }
                if (idx === rows.length - 1) {
                    setTimeout(function() {
                        dot.className = 'dot done';
                        logTerminal('POST /api/v1/query/report -> Final Structured Evidence Ready');
                    }, 600);
                }
            }, idx * 350);
        });
    }

    document.addEventListener('DOMContentLoaded', function() {
        initMap();
        initUI();
    });
})();
