pc.script.createLoadingScreen(function (app) {

    // ════════════════════════════════════════════════
    //  SEQUENCE
    //  1. Full black. Logo draws itself (worker canvas) with the loading %
    //     counting 20px below it.
    //  2. Logo finished + loading at 100% → the % vanishes, then the black
    //     opens in two bands revealing the LIVE scene (background video) —
    //     no poster image any more.
    //  3. Logo glides to the top-centre header position (same size/margin as
    //     OverlayManager), title + CTA fade in.
    //  4. CTA click → OverlayManager fades in underneath (its logo occupies
    //     the exact same pixels); once opaque the whole splash is cut in one
    //     frame — no blink on the handoff.
    // ════════════════════════════════════════════════
    // REPLAY: endingCredits.js sets this sessionStorage flag then reloads. On a
    // replay boot we re-run the exact same reveal, but SILENTLY — no % counter and
    // no minimum-black hold — so it jumps to the logo draw + bands without showing
    // the loading numbers again. Read + clear it once, so a later manual refresh is
    // a normal first-run load.
    var IS_REPLAY = false;
    try {
        IS_REPLAY = sessionStorage.getItem('edolusReplay') === '1';
        if (IS_REPLAY) sessionStorage.removeItem('edolusReplay');
    } catch (e) {}

    var REVEAL_MIN_MS    = IS_REPLAY ? 0 : 3000;  // min time fully black (0 on replay — the logo draw still gates the reveal)
    var FAKE_FILL_MS     = 3500;  // gentle time-based "fake" fill so the % never stalls
    // The % is a weighted blend of the two real phases of getting ready:
    //   asset preload  → 0 .. ASSET_SHARE
    //   scene warm-up  → ASSET_SHARE .. ~99%  (walk + settle, via 'warmup:progress')
    // so the counter moves continuously through the shader-compile work instead
    // of parking at a ceiling once assets (often cached) finish early.
    var ASSET_SHARE      = 0.80;
    var HOLD_BEFORE_OPEN = 0.99;  // hair below 100 until the app is truly ready
    var EASE_PER_FRAME   = 0.10;  // softness of the % easing into each value

    // After 'start', PlayCanvas does a last burst of main-thread work (a brief
    // microfreeze). We wait this long before opening the bands so the reveal
    // begins on a free thread AND the scene has rendered its first frames.
    var REVEAL_SETTLE_MS = 800;

    // Band opening — a single open, AFTER loading (the main thread is free by
    // then, and translateY runs on the compositor anyway, so no worker needed
    // for these — the worker below is for the LOGO, which still animates while
    // the preload hammers the main thread).
    var BAND_OPEN_MS = 1600;

    // Splash-logo geometry (loading phase, centre of screen).
    var SPLASH_LOGO_WIDTH = 430;      // px (max-width 85vw in CSS)
    var PROGRESS_GAP_PX   = 20;       // % readout top margin below the logo

    // Fallbacks if OverlayManager hasn't published its metrics yet.
    var OVERLAY_DEFAULTS = {
        width: 156, top: 40, widthMobile: 108, topMobile: 20,
        breakpoint: 768, fadeSeconds: 1.2
    };

    // ──────────────────────────────────────────────
    //  GENERAL SANS — prefer the font files imported in the project's fonts/
    //  folder (resolved from the asset registry at runtime); fall back to the
    //  Fontshare CDN if none are found in the build.
    // ──────────────────────────────────────────────
    var FONT_STYLE_ID = 'general-sans-faces';
    // All weights, so whichever script injects this shared face-set first
    // covers every consumer (loading title/subtitle, cards, world text).
    var FONT_FALLBACK_CSS =
        '@import url("https://api.fontshare.com/v2/css?f[]=general-sans@200,300,400,500,600,700&display=swap");';

    var injectGeneralSans = function () {
        if (document.getElementById(FONT_STYLE_ID)) return;
        var css = '';
        try {
            var matches = app.assets ? app.assets.filter(function (a) {
                var fn = (a.file && a.file.filename) || '';
                return a.file &&
                    /general[\s_-]?sans/i.test(a.name + ' ' + fn) &&
                    /\.(woff2?|ttf|otf)$/i.test(fn || a.name);
            }) : [];
            var faces = [];
            matches.forEach(function (a) {
                var n = (((a.file && a.file.filename) || a.name) + '').toLowerCase();
                var url = a.getFileUrl();
                if (!url) return;
                var w = /variable/.test(n) ? '100 900'
                      : /light/.test(n)    ? '300'
                      : /medium/.test(n)   ? '500'
                      : /semi/.test(n)     ? '600'
                      : /bold/.test(n)     ? '700' : '400';
                faces.push(
                    '@font-face{font-family:"General Sans";src:url("' + url + '");' +
                    'font-weight:' + w + ';font-style:' + (/italic/.test(n) ? 'italic' : 'normal') +
                    ';font-display:swap;}'
                );
            });
            css = faces.length ? faces.join('\n') : FONT_FALLBACK_CSS;
        } catch (err) {
            css = FONT_FALLBACK_CSS;
        }
        var st = document.createElement('style');
        st.id = FONT_STYLE_ID;
        st.textContent = css;
        document.head.appendChild(st);
    };

    // ──────────────────────────────────────────────
    // 1. CREATE THE SPLASH SCREEN HTML
    // ──────────────────────────────────────────────
    var showSplash = function () {
        var wrapper = document.createElement('div');
        wrapper.id = 'custom-splash-wrapper';
        // Suppress all splash transitions for the first paint(s) — see the
        // .preanim CSS note. Removed via a double rAF at the end of showSplash,
        // long before the reveal adds .loaded (Safari first-load flash guard).
        wrapper.classList.add('preanim');

        // Keep your environment check (needed for PlayCanvas editor)
        var isDev = window.location.href.includes('playcanvas') || window.location.href.includes('localhost');

        // Safely find the container. Dev → body. Production → wrapper,
        // falling back to body if the wrapper isn't ready yet.
        var container = isDev
            ? document.body
            : (document.getElementById('playcanvas-wrapper') || document.body);

        container.appendChild(wrapper);

        // LOGO — ABOVE the bands now: it draws itself on the pure black screen
        // while loading, and stays on top while the bands open beneath it.
        var logo = document.createElement('div');
        logo.id = 'custom-splash-logo';
        logo.innerHTML = [
            '<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 155 28" xml:space="preserve">',
            '<style type="text/css">.st0{fill:#FFFFFF;}</style>',
            '<g>',
            '<polygon class="st0" points="4.4,15.2 10.5,15.2 10.5,11.6 4.4,11.6 4.4,4.3 12,4.3 12,0.8 0.4,0.8 0.4,27.2 13.2,27.2 13.2,23.7 4.4,23.7 "/>',
            '<path class="st0" d="M42.3,6.7c-0.6-1.1-1.4-2.1-2.5-3c-1.1-0.9-2.5-1.6-4.1-2.1c-1.7-0.5-3.7-0.8-6-0.8h-6.4v26.5h6.6c2.7,0,5-0.4,6.8-1.2c1.8-0.8,3.3-1.8,4.4-3.2c1.1-1.3,1.8-2.8,2.3-4.4c0.5-1.6,0.7-3.3,0.7-4.9c0-1.1-0.1-2.2-0.4-3.4C43.4,8.9,42.9,7.8,42.3,6.7z M39.2,17.4c-0.3,1.2-0.8,2.2-1.6,3.2c-0.7,0.9-1.7,1.7-2.9,2.2c-1.2,0.6-2.7,0.8-4.5,0.8h-3V4.3h3c2.4,0,4.2,0.4,5.6,1.3c1.4,0.9,2.4,2,3,3.5c0.6,1.4,0.9,3,0.9,4.7C39.7,15,39.5,16.2,39.2,17.4z"/>',
            '<path class="st0" d="M153.9,17.2c-0.5-0.9-1-1.6-1.7-2.2c-0.8-0.7-1.7-1.4-2.7-1.9c-1-0.6-2-1.1-3-1.6c-1-0.6-1.8-1.2-2.4-1.8c-0.6-0.7-0.9-1.5-0.9-2.4c0-1.1,0.3-1.8,1-2.4c0.7-0.6,1.6-0.8,2.8-0.8c1,0,1.9,0.2,2.7,0.6c0.8,0.4,1.4,0.8,1.8,1.2l1.9-3.3c-0.3-0.3-0.7-0.6-1.3-0.9S151,1,150.2,0.8c-0.8-0.2-1.8-0.4-3-0.4c-1.4,0-2.8,0.3-4,0.8c-1.2,0.5-2.2,1.3-2.9,2.3c-0.7,1-1.1,2.4-1.1,4c0,1.6,0.3,3,1,4c0.7,1,1.5,1.9,2.4,2.5c0.8,0.5,1.7,1,2.6,1.4c0.9,0.4,1.8,0.8,2.5,1.3c0.8,0.5,1.4,1,1.9,1.6c0.5,0.6,0.7,1.4,0.7,2.3c0,1.1-0.4,1.9-1.3,2.5c-0.8,0.6-1.8,0.9-3.1,0.9c-0.7,0-1.5-0.1-2.2-0.4c-0.7-0.3-1.4-0.6-2-0.9c-0.6-0.4-1-0.6-1.2-0.9l-1.9,3.4c0.4,0.4,1,0.8,1.7,1.2c0.8,0.4,1.6,0.7,2.6,0.9c1,0.2,2,0.4,3,0.4c1.6,0,3-0.3,4.3-0.8c1.3-0.6,2.3-1.4,3-2.6c0.8-1.1,1.2-2.5,1.2-4.1C154.6,19,154.4,18,153.9,17.2z"/>',
            '<path class="st0" d="M124.7,16c0,2.9-0.4,4.9-1.3,6.1c-0.8,1.2-2.2,1.8-4.1,1.8c-1.9,0-3.3-0.6-4.1-1.8c-0.8-1.2-1.3-3.2-1.3-6V0.8h-4v15.2c0,2.4,0.3,4.5,0.9,6.2c0.6,1.8,1.6,3.1,3,4c1.4,0.9,3.2,1.4,5.5,1.4c3.2,0,5.6-1,7.1-3.1s2.3-4.9,2.3-8.6V0.8h-4.1V16z"/>',
            '<path class="st0" d="M76,4c-1.2-1.2-2.6-2.1-4.2-2.7c-0.8-0.3-1.6-0.5-2.4-0.7v3.8c0.7,0.2,1.3,0.4,1.9,0.8c1.4,0.7,2.5,1.9,3.3,3.3c0.8,1.5,1.2,3.3,1.2,5.5c0,2.2-0.4,4-1.2,5.5c-0.8,1.5-1.9,2.6-3.3,3.3c-1.4,0.8-2.9,1.2-4.5,1.2c-1.6,0-3.1-0.4-4.4-1.2c-1.4-0.8-2.5-1.9-3.3-3.3c-0.8-1.5-1.3-3.3-1.3-5.5c0-2.2,0.4-4,1.2-5.5c0.8-1.5,1.9-2.6,3.2-3.3c0.6-0.3,1.3-0.6,1.9-0.8V0.6c-0.8,0.2-1.7,0.4-2.5,0.7c-1.6,0.6-3,1.5-4.2,2.7c-1.2,1.2-2.2,2.6-2.9,4.3c-0.7,1.7-1,3.6-1,5.7c0,2.1,0.3,4,1,5.7c0.7,1.7,1.7,3.1,2.9,4.3c1.2,1.2,2.6,2.1,4.2,2.7c1.6,0.6,3.3,0.9,5.1,0.9c1.8,0,3.5-0.3,5.1-0.9c1.6-0.6,3-1.5,4.2-2.7c1.2-1.2,2.2-2.6,2.9-4.3C79.6,18,80,16.1,80,14c0-2.1-0.4-4-1.1-5.7C78.2,6.6,77.2,5.1,76,4z"/>',
            '<polygon class="st0" points="94.4,0.8 90.3,0.8 90.3,27.2 102,27.2 102,23.6 94.4,23.6 "/>',
            '</g>',
            '</svg>'
        ].join('');
        wrapper.appendChild(logo);

        // Hide the glyph fills from the very first frame so the logo never
        // flashes fully-formed before the draw-on animation takes over.
        var glyphs = logo.querySelectorAll('svg path, svg polygon, svg polyline');
        for (var gi = 0; gi < glyphs.length; gi++) {
            glyphs[gi].style.fillOpacity = '0';
        }

        // BLACK BANDS (top + bottom). They stay CLOSED (a uniform black screen)
        // for the whole load, then open in two parts revealing the live scene.
        var bandTop = document.createElement('div');
        bandTop.id = 'custom-band-top';
        wrapper.appendChild(bandTop);

        var bandBottom = document.createElement('div');
        bandBottom.id = 'custom-band-bottom';
        wrapper.appendChild(bandBottom);

        // PROGRESS READOUT — 20px below the logo (positioned by JS once the
        // logo has a layout box; see placeProgress).
        var progressText = document.createElement('div');
        progressText.id = 'custom-progress-text';
        // One masked COLUMN per digit (hundreds / tens / units) plus a STATIC "%"
        // that never moves. Each digit rolls independently — only the ones that
        // change flip. The reel spans inside each column are created on first tick.
        progressText.innerHTML =
            '<span class="roll-col"></span>' +
            '<span class="roll-col"></span>' +
            '<span class="roll-col"></span>' +
            '<span class="pct-sign">%</span>';
        // On a replay boot, keep the counter hidden — just the black + logo draw.
        if (IS_REPLAY) progressText.style.display = 'none';
        wrapper.appendChild(progressText);

        // ──────────────────────────────────────────
        //  TITLE GROUP + CTA  (revealed after the bands open)
        // ──────────────────────────────────────────
        var titleGroup = document.createElement('div');
        titleGroup.id = 'custom-title-group';

        var title = document.createElement('div');
        title.id = 'custom-title';
        title.textContent = 'INTELLIGENCE\nAT PLANETARY SCALE';

        var subtitle = document.createElement('div');
        subtitle.id = 'custom-subtitle';
        subtitle.textContent = 'Orchestrating the foundation of artificial intelligence\nfrom the edge of the atmosphere.';

        titleGroup.appendChild(title);
        titleGroup.appendChild(subtitle);
        wrapper.appendChild(titleGroup);

        // Bottom notice: headphones icon + a subtitle-styled line, pinned to the BOTTOM
        // with the same inset the Edolus logo sits from the TOP (OVERLAY_DEFAULTS.top).
        // Fades in / out with the title group (see the .loaded / .clicked rules).
        var headphones = document.createElement('div');
        headphones.id = 'custom-headphones';
        headphones.innerHTML =
            '<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
            '<path d="M39.5,18.1c0.4-0.6,0.5-1.5,0.2-2.5l0-0.2c-0.9-3.8-3.6-15.2-19.6-15.3v0c0,0,0,0,0,0s0,0,0,0v0C4,0.2,1.3,11.7,0.4,15.4l0,0.2c-0.3,1.1-0.2,1.9,0.2,2.5c0.1,0.1,0.1,0.1,0.2,0.2c-0.7,5.9,0.7,11.7,0.8,12l0.1,0c1.3,5,2.8,7.9,4.7,9c0.7,0.4,1.4,0.5,2,0.5c0.3,0,0.6,0,0.9-0.1c0.4,0.1,0.9,0.1,1.4,0.1c1.3,0,2.9-0.3,4.4-1.2c2.1-1.5,3-4.7,2.3-8.6c-0.7-3.8-3.1-8.3-6.8-9.1c-2.8-0.6-4.9,0.6-6,1.5l-0.3-3.4c0,0,0.1,0,0.1-0.1c0.1-0.1,0.2-0.3,0.3-0.5C5.2,12.5,8.4,5.2,20,5.2c11.6,0,14.8,7.3,15.5,13.3c0,0.2,0.1,0.3,0.3,0.5c0,0,0.1,0,0.1,0.1l-0.3,3.4c-1.1-0.9-3.2-2.1-6-1.5c-3.6,0.8-6.1,5.3-6.8,9.1c-0.7,3.9,0.1,7.1,2.3,8.6c1.4,1,3.1,1.2,4.4,1.2c0.5,0,1,0,1.4-0.1c0.3,0,0.6,0.1,0.9,0.1c0.7,0,1.3-0.1,2-0.5c1.9-1.1,3.4-4,4.7-9l0.1,0c0.1-0.2,1.5-6.1,0.8-12C39.4,18.3,39.4,18.2,39.5,18.1z M1.7,19c0.1,0,0.1,0,0.2,0l0.2,0c0.4,0.1,0.7,0.1,1,0.1c0,0,0,0,0,0c0,0,0.1,0,0.1,0l0.3,3.3c0,0.2-0.1,0.9,0.3,1.3c0.2,0.2,0.5,0.3,0.8,0.3c2.2-0.2,5.1,0.5,6.6,3.4c1.4,2.5,2.2,6.4,1.1,9c-0.3,0.7-0.7,1.2-1.3,1.6c1-1.6,1.2-4.3,0.3-7c-0.8-2.5-2.3-4.5-4-5.4c-0.9-0.4-1.7-0.5-2.5-0.3c-1.4,0.5-2.3,2-2.5,3.9C2,27.6,1.2,23.3,1.7,19z M9,38.5c-0.2,0.1-0.4,0.1-0.5,0.1c0,0,0,0,0,0l0,0c-0.4,0-0.8-0.1-1.2-0.3c-1.4-0.7-2.7-2.5-3.5-4.8c-1.1-3.3-0.5-6.6,1.3-7.2c0.2-0.1,0.4-0.1,0.6-0.1c0.4,0,0.7,0.1,1.1,0.3c1.4,0.7,2.7,2.5,3.5,4.8C11.4,34.6,10.8,37.9,9,38.5z M10.2,21.9c3.2,0.7,5.4,4.8,6,8.3c0.6,3.5-0.1,6.4-1.8,7.6c-0.9,0.6-1.8,0.9-2.7,1c0.7-0.5,1.2-1.1,1.5-2c1.2-2.9,0.3-7.1-1.2-9.8c-1.6-3-4.5-3.9-6.9-3.9C6.1,22.4,7.9,21.4,10.2,21.9z M20.1,4.2L20.1,4.2C20,4.2,20,4.2,20.1,4.2C20,4.2,20,4.2,20.1,4.2L20.1,4.2c-12.8,0-15.8,8.7-16.5,14c-0.1,0-0.2,0-0.4,0c-0.1,0-0.3,0-0.4-0.1c-0.2,0-0.7-0.2-1.1-0.2c-0.1-0.1-0.2-0.1-0.3-0.3c-0.3-0.4-0.3-1-0.1-1.8l0-0.2C2.2,12.1,4.8,1.2,20,1.2c15.3,0,17.8,10.9,18.7,14.5l0,0.2c0.2,0.8,0.1,1.4-0.1,1.8c-0.1,0.1-0.2,0.2-0.3,0.3c-0.4,0.1-0.9,0.2-1.1,0.2c-0.1,0-0.3,0.1-0.4,0.1c-0.1,0-0.3,0-0.4,0C35.9,12.9,32.8,4.2,20.1,4.2z M23.8,30.2c0.7-3.5,2.8-7.6,6-8.3c2.3-0.5,4.1,0.5,5.1,1.2c-2.4,0-5.3,1-6.9,3.9c-1.5,2.7-2.4,7-1.2,9.8c0.3,0.8,0.9,1.5,1.5,2c-0.9-0.1-1.8-0.4-2.7-1C23.9,36.6,23.2,33.7,23.8,30.2z M36.3,33.6c-0.7,2.2-2,4.1-3.5,4.8c-0.4,0.2-0.8,0.3-1.2,0.3l0,0c0,0,0,0,0,0c-0.2,0-0.4,0-0.5-0.1c-1.8-0.6-2.4-3.9-1.3-7.2c0.7-2.2,2-4.1,3.5-4.8c0.6-0.3,1.2-0.4,1.7-0.2C36.7,26.9,37.3,30.3,36.3,33.6z M37.7,29.3c-0.2-2-1.1-3.5-2.5-3.9c-0.8-0.3-1.6-0.2-2.5,0.3c-1.7,0.8-3.2,2.8-4,5.4c-0.9,2.8-0.7,5.4,0.3,7c-0.5-0.4-1-0.9-1.3-1.6c-1.1-2.6-0.2-6.4,1.1-9c1.4-2.5,3.8-3.4,5.9-3.4c0.3,0,0.5,0,0.8,0c0.3,0,0.6-0.1,0.8-0.3c0.4-0.4,0.3-1.1,0.3-1.3l0.3-3.3c0,0,0.1,0,0.1,0c0,0,0,0,0,0c0.3,0,0.6-0.1,1-0.1l0.2,0c0,0,0.1,0,0.2,0C38.8,23.3,38.1,27.6,37.7,29.3z"/>' +
            '</svg>' +
            '<div class="hp-text">Experience with headphones</div>';
        wrapper.appendChild(headphones);

        var cta = document.createElement('div');
        cta.id = 'custom-cta';
        // Text in its own span so it can blink on hover without flashing the white
        // button background (which a blink on #custom-cta itself would).
        cta.innerHTML = '<span id="cta-label">INITIATE SYSTEM</span>';
        // The button is revealed by animateCta (a construct-in: 1px line → expand →
        // scramble), NOT the CSS opacity fade — keep it hidden + non-interactive until
        // then (inline styles beat the .loaded rule).
        cta.style.opacity = '0';
        cta.style.pointerEvents = 'none';
        // Fixed-height SLOT reserves the button's final vertical space and flex-centres
        // it, so the height animation (1px → full) opens the button from its CENTRE
        // without reflowing the (translate-centred) title group above it.
        var ctaSlot = document.createElement('div');
        ctaSlot.id = 'custom-cta-slot';
        ctaSlot.appendChild(cta);
        titleGroup.appendChild(ctaSlot);

        // CTA HOVER → tell the live scene to slowly push the background video in
        // (videoEarth.js listens for these). Reverts on leave (and on click, which
        // also fires 'experienceStart' → videoEarth eases back).
        cta.addEventListener('mouseenter', function () {
            window.dispatchEvent(new CustomEvent('cta:hover'));
        });
        cta.addEventListener('mouseleave', function () {
            window.dispatchEvent(new CustomEvent('cta:unhover'));
        });

        // CTA CLICK → seamless logo handoff + dismiss splash.
        // The splash logo does NOT fade: OverlayManager's logo fades in at the
        // exact same position/size underneath it (experienceStart), and once
        // it is fully opaque we cut the whole splash in a single frame — the
        // two logos are pixel-identical, so nothing blinks.
        cta.addEventListener('click', function () {
            cta.classList.add('blink');
            wrapper.classList.add('clicked');   // fades title/CTA + the 50% dim

            // Fire the ChromaticTransition post-effect as the click flourish —
            // chromaticTransition.js listens for this on window (DOM-side twin
            // of its EventBus trigger). Speed = its "Play Duration" attribute;
            // pass { detail: { duration: s } } here to override per-click.
            window.dispatchEvent(new CustomEvent('chromatic:play'));

            window.dispatchEvent(new CustomEvent('experienceStart'));

            var M = window.__overlayLogoMetrics || OVERLAY_DEFAULTS;
            var overlayFadeMs = (M.fadeSeconds || OVERLAY_DEFAULTS.fadeSeconds) * 1000;
            // Wait for the LONGEST of: overlay fade-in (logo swap), title fade
            // (1.5s) — then remove everything at once.
            setTimeout(function () {
                window.removeEventListener('resize', onResize);
                if (wrapper.parentNode) wrapper.parentNode.removeChild(wrapper);
            }, Math.max(overlayFadeMs, 1500) + 200);
        });

        // Everything is inserted at its resting opacity:0 with transitions
        // killed (.preanim). Let it paint once, then release the guard so the
        // real .loaded / .clicked fades work normally. Double rAF guarantees a
        // frame with transitions disabled has actually painted first.
        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                wrapper.classList.remove('preanim');
            });
        });
    };

    // ──────────────────────────────────────────────
    // 2. STATE
    // ──────────────────────────────────────────────
    var realProgress = 0;      // actual preload progress (0..1)
    var warmupProgress = 0;    // scene warm-up progress (0..1, from SceneManager)
    var displayProgress = 0;   // smoothed value we actually render
    var loadingComplete = false;
    var logoAnimDone = false;  // set by the worker when the last letter lands
    var revealed = false;
    var logoDrawn = false;     // guards the one-shot letter-draw animation
    var logoDocked = false;    // logo has moved to its header position
    var startTime = null;
    var lastTime = null;
    var rafId = null;
    var gateFired = false;
    var rollCols = null;       // per-digit columns [{top,bot,rolling}] hundreds→units
    // Reel state — the number rolls in discrete steps that LAND and briefly hold on
    // a whole number (so a crisp number is always readable while the % climbs fast).
    // Each digit has its OWN column and only rolls when THAT digit changes, so e.g.
    // 10→19 flips just the units while the "1" and "%" stay put (see renderProgress).
    var ROLL_MS       = 200;   // one step's slide time (from → to)
    var ROLL_HOLD_MS  = 70;    // pause on the landed number before the next step
    var rollFrom      = 0;     // number currently resting in the band
    var rollTo        = 0;     // number the active step is sliding toward
    var rollStart     = 0;     // timestamp the active step began
    var rollActive    = false; // a step is mid-slide
    var rollHoldUntil = 0;     // don't start the next step before this time

    // ──────────────────────────────────────────────
    //  LOGO DRAW-ON ANIMATION  (OffscreenCanvas + Web Worker — jank-proof)
    //  The PlayCanvas preload hogs the MAIN thread, so any animation ticking
    //  there (GSAP, SVG, JS rAF) micro-freezes while loading. The reveal is
    //  rendered on an OffscreenCanvas driven by a Web Worker on its OWN
    //  thread — it keeps painting at full speed no matter how busy the main
    //  thread gets. (This worker is for the LOGO — the bands need no worker,
    //  they only move after loading is done.)
    //
    //  Reveal style: each letter is revealed by a soft-edged left→right WIPE
    //  (clipped to the glyph), letters staggered left→right so the whole
    //  wordmark sweeps on in one continuous, premium motion. Each letter
    //  completes in a single stroke — no mid-letter stalls.
    //
    //  Fallback: if OffscreenCanvas is unavailable, staggered fill fades.
    // ──────────────────────────────────────────────
    var DRAW_MS      = 1100;  // wipe time for a single letter
    var STAGGER_MS   = 220;   // gap between consecutive letters starting
    var EDGE_FEATHER = 14;    // soft leading-edge width (viewBox units)

    // Worker source. Runs OFF the main thread, so PlayCanvas loading can't
    // stall it. Per glyph: clip to the letter, fill a solid region plus a
    // feathered gradient strip at the leading edge → soft wipe.
    var WORKER_SRC = [
        'self.onmessage = function (e) {',
        '  var d = e.data;',
        '  var canvas = d.canvas;',
        '  var ctx = canvas.getContext("2d");',
        '  var sx = canvas.width / d.vbw;',
        '  var sy = canvas.height / d.vbh;',
        '  var paths = d.paths.map(function (s) { return new Path2D(s); });',
        '  var bbs = d.bboxes, delays = d.delays, DRAW = d.drawMs, F = d.feather;',
        '  var n = paths.length, start = null;',
        // easeInOutCubic — one smooth accelerate/decelerate per letter
        '  function ease(t){ t = t<0?0:t>1?1:t; return t<0.5 ? 4*t*t*t : 1 - Math.pow(-2*t+2,3)/2; }',
        '  var raf = self.requestAnimationFrame ? self.requestAnimationFrame.bind(self)',
        '          : function (cb){ setTimeout(function(){ cb(performance.now()); }, 16); };',
        '  function frame(now){',
        '    if (start === null) start = now;',
        '    var t = now - start, done = true, i;',
        '    ctx.setTransform(1,0,0,1,0,0);',
        '    ctx.clearRect(0,0,canvas.width,canvas.height);',
        '    ctx.setTransform(sx,0,0,sy,0,0);',
        '    for (i = 0; i < n; i++) {',
        '      var p = (t - delays[i]) / DRAW;',
        '      if (p < 1) done = false;',
        '      if (p <= 0) continue;',
        '      var bb = bbs[i];',
        '      ctx.save();',
        '      ctx.clip(paths[i]);',
        '      if (p >= 1) {',
        '        ctx.fillStyle = "#ffffff";',
        '        ctx.fill(paths[i]);',
        '      } else {',
        '        var edge  = bb.x + ease(p) * (bb.w + F);',   // leading edge x
        '        var solid = edge - F;',                       // fully-opaque up to here
        '        ctx.fillStyle = "#ffffff";',
        '        if (solid > bb.x) ctx.fillRect(bb.x - 0.5, bb.y - 0.5, solid - bb.x + 0.5, bb.h + 1);',
        '        var g = ctx.createLinearGradient(solid, 0, edge, 0);',
        '        g.addColorStop(0, "rgba(255,255,255,1)");',
        '        g.addColorStop(1, "rgba(255,255,255,0)");',
        '        ctx.fillStyle = g;',
        '        ctx.fillRect(solid, bb.y - 0.5, F, bb.h + 1);',
        '      }',
        '      ctx.restore();',
        '    }',
        '    if (done) { self.postMessage({ done: true }); return; }',
        '    raf(frame);',
        '  }',
        '  raf(frame);',
        '};'
    ].join('\n');

    // polygon/polyline have no "d"; turn their points into a path string.
    var shapeToPathD = function (shape) {
        var tag = shape.tagName.toLowerCase();
        if (tag === 'polygon' || tag === 'polyline') {
            var pts = (shape.getAttribute('points') || '').trim().split(/[\s,]+/);
            var d = '';
            for (var i = 0; i + 1 < pts.length; i += 2) {
                d += (i === 0 ? 'M' : 'L') + pts[i] + ' ' + pts[i + 1] + ' ';
            }
            if (tag === 'polygon') d += 'Z';
            return d;
        }
        return shape.getAttribute('d') || '';
    };

    var fallbackReveal = function (shapes) {
        // No OffscreenCanvas support → staggered left→right fill fades.
        shapes.forEach(function (shape, i) {
            shape.style.transition = 'fill-opacity 0.9s ease ' + (i * 0.15) + 's';
            shape.style.fillOpacity = '1';
        });
        setTimeout(function () { logoAnimDone = true; },
            shapes.length * 150 + 900 + 100);
    };

    var animateLogo = function () {
        if (logoDrawn) return;
        var logo = document.getElementById('custom-splash-logo');
        if (!logo) return;
        var svg = logo.querySelector('svg');
        if (!svg) return;

        var shapes = Array.prototype.slice.call(
            svg.querySelectorAll('path, polygon, polyline')
        );
        if (!shapes.length) return;

        logoDrawn = true;

        // Reading order (left → right) regardless of DOM order — the wipe
        // sweeps across the wordmark in one continuous direction.
        shapes.sort(function (a, b) {
            return a.getBBox().x - b.getBBox().x;
        });

        var paths = shapes.map(shapeToPathD);
        var bboxes = shapes.map(function (shape) {
            var b = shape.getBBox();
            return { x: b.x, y: b.y, w: b.width, h: b.height };
        });
        var delays = shapes.map(function (_, i) { return i * STAGGER_MS; });

        // viewBox dims for the worker's coordinate mapping.
        var vbParts = (svg.getAttribute('viewBox') || '0 0 155 28').split(/[\s,]+/);
        var vbw = parseFloat(vbParts[2]) || 155;
        var vbh = parseFloat(vbParts[3]) || 28;

        // Canvas overlay, sized to the logo box at device resolution.
        var rect = svg.getBoundingClientRect();
        var dpr = window.devicePixelRatio || 1;
        var canvas = document.createElement('canvas');
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.width = Math.max(1, Math.round(rect.width * dpr));
        canvas.height = Math.max(1, Math.round(rect.height * dpr));
        logo.appendChild(canvas);

        // Hand the canvas to a worker thread; it animates independently of
        // whatever the main thread is doing (PlayCanvas loading included).
        if (!canvas.transferControlToOffscreen || typeof Worker === 'undefined') {
            canvas.parentNode.removeChild(canvas);
            fallbackReveal(shapes);
            return;
        }

        try {
            var offscreen = canvas.transferControlToOffscreen();
            var blob = new Blob([WORKER_SRC], { type: 'application/javascript' });
            var worker = new Worker(URL.createObjectURL(blob));
            worker.onmessage = function (e) {
                if (!e.data || !e.data.done) return;
                logoAnimDone = true;
                // Swap to the crisp vector: turn the SVG fills on, then drop
                // the canvas next frame (identical pixels → invisible swap).
                // The vector scales cleanly during the later move-to-header.
                shapes.forEach(function (s) { s.style.fillOpacity = '1'; });
                requestAnimationFrame(function () {
                    if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
                });
                worker.terminate();
            };
            worker.postMessage({
                canvas: offscreen,
                paths: paths,
                bboxes: bboxes,
                delays: delays,
                drawMs: DRAW_MS,
                feather: EDGE_FEATHER,
                vbw: vbw,
                vbh: vbh
            }, [offscreen]);
            // Belt & braces: if the worker's done message is ever lost, don't
            // deadlock the reveal gate.
            setTimeout(function () { logoAnimDone = true; },
                (shapes.length - 1) * STAGGER_MS + DRAW_MS + 600);
        } catch (err) {
            if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
            fallbackReveal(shapes);
        }
    };

    // ──────────────────────────────────────────────
    // 3. PROGRESS READOUT (below the logo)
    // ──────────────────────────────────────────────
    var setProgress = function (value) {
        realProgress = Math.min(1, Math.max(0, value));
    };

    // STEPPED PER-DIGIT ODOMETER — one masked column per digit (the "%" beside
    // them is static). The % climbs ~20 numbers/second, far too fast for a
    // continuous reel to ever rest on a digit (it just smears into moving lines),
    // so the whole number rolls in DISCRETE steps that LAND and hold briefly — a
    // crisp number is always readable. Within each step, ONLY the digit columns
    // that actually change roll; shared leading digits (and the "%") stay fixed.
    // e.g. 10→19 flips just the units; 19→20 flips units AND tens. When the % is
    // climbing fast a step jumps straight to the current target (skipping the
    // in-betweens); near the end the steps become single increments onto 100.
    var NUM_COLS = 3;   // hundreds, tens, units — enough for "100"
    var easeOutCubic = function (t) { return 1 - Math.pow(1 - t, 3); };

    // Right-aligned digits with blank leading slots: 5→["","","5"], 19→["","1","9"].
    var digitsOf = function (n) {
        var s = String(n);
        var out = ['', '', ''];
        for (var i = 0; i < s.length; i++) out[NUM_COLS - s.length + i] = s.charAt(i);
        return out;
    };

    var renderProgress = function (now) {
        if (now == null) {
            now = (typeof performance !== 'undefined' && performance.now)
                ? performance.now() : Date.now();
        }
        var band = document.getElementById('custom-progress-text');
        if (!band) return;

        if (!rollCols) {
            var colEls = band.querySelectorAll('.roll-col');
            if (!colEls.length) return;
            rollCols = [];
            for (var c = 0; c < colEls.length; c++) {
                var top = document.createElement('span'); top.className = 'roll-num';
                var bot = document.createElement('span'); bot.className = 'roll-num';
                colEls[c].appendChild(top);
                colEls[c].appendChild(bot);
                bot.style.transform = 'translateY(100%)';
                rollCols.push({ top: top, bot: bot, rolling: false });
            }
            var d0 = digitsOf(rollTo);
            for (var k = 0; k < rollCols.length; k++) rollCols[k].top.textContent = d0[k];
        }

        var target = Math.round(displayProgress * 100);
        if (target < 0) target = 0; else if (target > 100) target = 100;

        // Idle and past the hold, but behind the target → begin the next step.
        // Set each column rolling ONLY if its digit differs from → to.
        if (!rollActive && now >= rollHoldUntil && target !== rollTo) {
            rollFrom = rollTo;
            rollTo = target;
            rollStart = now;
            rollActive = true;
            var f = digitsOf(rollFrom), t = digitsOf(rollTo);
            for (var i = 0; i < rollCols.length; i++) {
                var col = rollCols[i];
                col.rolling = (f[i] !== t[i]);
                if (col.rolling) {
                    col.top.textContent = f[i];     // outgoing digit
                    col.bot.textContent = t[i];     // incoming digit, waiting below
                    col.top.style.transform = 'translateY(0)';
                    col.bot.style.transform = 'translateY(100%)';
                } else {
                    col.top.textContent = t[i];     // unchanged → stays put
                    col.top.style.transform = 'translateY(0)';
                    col.bot.style.transform = 'translateY(100%)';
                }
            }
        }

        if (rollActive) {
            var p = (now - rollStart) / ROLL_MS;
            if (p >= 1) p = 1;
            var e = easeOutCubic(p);
            for (var j = 0; j < rollCols.length; j++) {
                var cj = rollCols[j];
                if (!cj.rolling) continue;   // untouched digits don't move
                cj.top.style.transform = 'translateY(' + (-e * 100) + '%)';
                cj.bot.style.transform = 'translateY(' + ((1 - e) * 100) + '%)';
            }
            if (p >= 1) {
                // Landed — promote each rolled digit to its resting slot and hold.
                rollActive = false;
                rollHoldUntil = now + ROLL_HOLD_MS;
                for (var m = 0; m < rollCols.length; m++) {
                    var cm = rollCols[m];
                    if (!cm.rolling) continue;
                    cm.top.textContent = cm.bot.textContent;
                    cm.top.style.transform = 'translateY(0)';
                    cm.bot.style.transform = 'translateY(100%)';
                    cm.rolling = false;
                }
            }
        }
    };

    // Pin the % readout 20px under the logo's live layout box.
    var placeProgress = function () {
        var logo = document.getElementById('custom-splash-logo');
        var text = document.getElementById('custom-progress-text');
        if (!logo || !text) return;
        var r = logo.getBoundingClientRect();
        text.style.top = (r.bottom + PROGRESS_GAP_PX) + 'px';
    };

    // ──────────────────────────────────────────────
    // 4. LOGO → HEADER (top-centre, OverlayManager's size + top margin)
    // ──────────────────────────────────────────────
    var overlayMetrics = function () {
        var M = window.__overlayLogoMetrics || OVERLAY_DEFAULTS;
        var mobile = window.innerWidth <= (M.breakpoint || OVERLAY_DEFAULTS.breakpoint);
        return {
            width: mobile ? (M.widthMobile || OVERLAY_DEFAULTS.widthMobile)
                          : (M.width || OVERLAY_DEFAULTS.width),
            top:   mobile ? (M.topMobile != null ? M.topMobile : OVERLAY_DEFAULTS.topMobile)
                          : (M.top != null ? M.top : OVERLAY_DEFAULTS.top)
        };
    };

    var dockLogo = function (animate) {
        var logo = document.getElementById('custom-splash-logo');
        if (!logo) return;
        var m = overlayMetrics();

        // Snap (no animation) — used on resize once docked, to keep matching the overlay.
        if (!animate) {
            logo.style.transition = 'none';
            logo.style.top = m.top + 'px';
            logo.style.width = m.width + 'px';
            logo.style.transform = 'translate(-50%, 0) translateZ(0)';
            logoDocked = true;
            return;
        }

        // FLIP: move UP + scale DOWN on ONE composited transform so they stay perfectly in
        // sync. The old version animated `width` (a layout property) alongside the move —
        // fine on PC, but on mobile the per-frame reflow lagged the move, so the logo slid
        // up first and scaled late. Here the FINAL layout is applied instantly (so the
        // OverlayManager handoff stays pixel-perfect) and ONLY `transform` animates, from
        // an inverted (old: big + centred) pose back to rest.
        var first = logo.getBoundingClientRect();                 // FIRST — big + centred

        logo.style.transition = 'none';                           // LAST — the final layout
        logo.style.top = m.top + 'px';
        logo.style.width = m.width + 'px';
        logo.style.transformOrigin = 'center center';
        logo.style.transform = 'translate(-50%, 0) translateZ(0)';
        var last = logo.getBoundingClientRect();                  // (forces layout)

        // INVERT — a transform that maps the final rect back onto the old one.
        var s  = last.width > 0 ? first.width / last.width : 1;   // scale up to the old size
        var dx = (first.left + first.width  / 2) - (last.left + last.width  / 2);
        var dy = (first.top  + first.height / 2) - (last.top  + last.height / 2);
        logo.style.transform =
            'translate(-50%, 0) translate(' + dx + 'px, ' + dy + 'px) scale(' + s + ') translateZ(0)';

        // PLAY — flush the inverted pose as the transition's start, then ease the transform
        // back to rest. transform-only → composited, smooth, and move + scale are one prop.
        void logo.offsetWidth;                                    // reflow → invert is the baseline
        logo.style.transition = 'transform 1s cubic-bezier(0.25,0.1,0.25,1)';
        logo.style.transform = 'translate(-50%, 0) translateZ(0)';

        logoDocked = true;
    };

    var onResize = function () {
        if (logoDocked) dockLogo(false);   // keep matching the overlay's media query
        else placeProgress();              // keep the % pinned under the logo
    };

    // ──────────────────────────────────────────────
    // 5. RAF LOOP — smooth %, one reveal gate
    // ──────────────────────────────────────────────
    var tick = function (now) {
        if (startTime === null) {
            startTime = now; lastTime = now;
            placeProgress();
            animateLogo();     // runs on the worker thread — safe during preload
        }
        var dt = now - lastTime;
        lastTime = now;

        var elapsed = now - startTime;

        // Asset phase: whichever is further along — real loading or the gentle
        // timer. Then blend in the warm-up phase so the counter keeps moving
        // through the shader-compile walk + settle (see ASSET_SHARE).
        var timeFloor = elapsed / FAKE_FILL_MS;
        var assetP = Math.max(realProgress, timeFloor);
        if (assetP > 1) assetP = 1;
        var effective = ASSET_SHARE * assetP + (1 - ASSET_SHARE) * warmupProgress;
        var cap = loadingComplete ? 1 : HOLD_BEFORE_OPEN;
        if (effective > cap) effective = cap;

        displayProgress += (effective - displayProgress) *
            (1 - Math.pow(1 - EASE_PER_FRAME, dt / 16.6667));
        renderProgress(now);

        // THE GATE: app ready + logo fully drawn + minimum black time.
        // (The elapsed>8s guard means a lost worker can never deadlock us.)
        if (loadingComplete &&
            (logoAnimDone || elapsed > 8000) &&
            elapsed >= REVEAL_MIN_MS) {
            fireGate();
            rafId = null;
            return;
        }

        rafId = requestAnimationFrame(tick);
    };

    var fireGate = function () {
        if (gateFired) return;
        gateFired = true;

        // Roll the readout up to 100 and let it land. The main tick loop is
        // about to stop, so drive the reel with its own short rAF until it has
        // settled on 100 (clearing any pending hold so the last step starts now).
        displayProgress = 1;
        rollHoldUntil = 0;
        var finishRoll = function () {
            var t = (typeof performance !== 'undefined' && performance.now)
                ? performance.now() : Date.now();
            renderProgress(t);
            if (!(rollTo === 100 && !rollActive)) requestAnimationFrame(finishRoll);
        };
        finishRoll();

        // …then it just disappears (no fade), and the black opens.
        setTimeout(function () {
            var text = document.getElementById('custom-progress-text');
            if (text) text.style.display = 'none';
        }, 400);

        setTimeout(openBands, REVEAL_SETTLE_MS);
    };

    // The black screen opens in two parts, revealing the LIVE scene (the
    // background video is already playing behind the splash by now).
    var openBands = function () {
        var w = document.getElementById('custom-splash-wrapper');
        if (!w) return;
        w.classList.add('bands-open');
        setTimeout(onRevealComplete, BAND_OPEN_MS + 100);
    };

    // ──────────────────────────────────────────────
    // 6. REVEAL COMPLETE → logo to header, title phase
    // ──────────────────────────────────────────────
    var onRevealComplete = function () {
        if (revealed) return;
        revealed = true;

        var splash = document.getElementById('custom-splash-wrapper');
        if (!splash) return;

        splash.classList.add('loaded');
        dockLogo(true);                       // glide to top-centre header spot
        setTimeout(animateTitle, 200);
        setTimeout(animateCta, 900);          // build the button in (line → expand → scramble)
    };

    // ──────────────────────────────────────────────
    // GSAP LETTER ANIMATION (title phase)
    // ──────────────────────────────────────────────
    var animateTitle = function () {
        var title = document.getElementById('custom-title');
        if (!title || !window.gsap) return;

        var text = title.textContent;
        title.textContent = '';

        var letters = [];

        for (var i = 0; i < text.length; i++) {
            if (text[i] === '\n') {
                title.appendChild(document.createElement('br'));
                continue;
            }
            var span = document.createElement('span');
            span.textContent = text[i];
            span.style.opacity = '0';
            title.appendChild(span);
            letters.push(span);
        }

        var shuffled = letters.slice().sort(function () {
            return Math.random() - 0.5;
        });

        // Original pacing (1.2s per letter, 0.5s lead-in, 0.06s stagger) — kept
        // as-is. The only enhancement: each letter de-blurs as it fades in, the
        // blur melting away in lockstep with the opacity for a premium,
        // developing-into-focus feel.
        shuffled.forEach(function (letter, index) {
            gsap.fromTo(letter,
                { opacity: 0, filter: 'blur(12px)' },
                {
                    opacity: 1,
                    filter: 'blur(0px)',
                    duration: 1.2,
                    delay: 0.5 + (index * 0.06),
                    ease: "expo.out"
                }
            );
        });
    };

    // ──────────────────────────────────────────────
    // CTA CONSTRUCT-IN (title phase)
    //   1. A 1px-tall white line (the button collapsed) sits at the button's final
    //      CENTRE — the four corner squares start close together on each side.
    //   2. The button HEIGHT expands to full; the corners follow the edges outward.
    //   3. At 50% of the final height, the "INITIATE SYSTEM" label SCRAMBLES in.
    // Extremely quick. The slot reserves the layout height so nothing above reflows.
    // ──────────────────────────────────────────────
    var ctaBuilt = false;
    var animateCta = function () {
        if (ctaBuilt) return;
        var cta   = document.getElementById('custom-cta');
        var slot  = document.getElementById('custom-cta-slot');
        var label = document.getElementById('cta-label');
        if (!cta || !slot || !window.gsap) return;
        ctaBuilt = true;

        var EXPAND_S  = 0.18;   // button open time (s) — extremely quick
        var SCR_SPEED = 1.5;    // label scramble speed (higher = quicker decode)

        // Measure the natural (final) width + height, then pin the SLOT height (reserve
        // layout space, no reflow) and LOCK the button width — only the height animates,
        // so the button opens purely vertically (blanking the label must not shrink it).
        var finalH = cta.offsetHeight;
        var finalW = cta.offsetWidth;
        slot.style.height = finalH + 'px';

        // Take over the reveal: show INSTANTLY (the button IS the 1px line now) with no
        // opacity fade; border-box so width/height == the visible box (padding inside);
        // overflow visible so the corner squares (::before, outside the box) never clip.
        cta.style.transition = 'padding 0.3s ease';   // keep hover padding, drop the opacity fade
        cta.style.boxSizing  = 'border-box';
        cta.style.width      = finalW + 'px';          // constant width — height is the only axis that grows
        cta.style.overflow   = 'visible';
        cta.style.opacity    = '1';

        // Label blank until 50% height, then it scrambles in.
        var scr = (window.TextScramble && label) ? new TextScramble(label, SCR_SPEED) : null;
        if (label) label.textContent = '';

        gsap.set(cta, { height: 1 });   // collapse to a 1px line (slot flex-centres it)
        var scrambled = false;
        gsap.to(cta, {
            height: finalH,
            duration: EXPAND_S,
            ease: 'power3.out',
            onUpdate: function () {
                // Fire the scramble once the button crosses 50% of its final height.
                if (!scrambled && parseFloat(cta.style.height) >= finalH * 0.5) {
                    scrambled = true;
                    if (scr) scr.setText('INITIATE SYSTEM');
                    else if (label) label.textContent = 'INITIATE SYSTEM';
                }
            },
            onComplete: function () {
                // Release the pinned sizes → natural (content-driven, responsive) again,
                // and make the button interactive.
                gsap.set(cta, { clearProps: 'height' });
                cta.style.width = '';
                slot.style.height = '';
                cta.style.pointerEvents = 'auto';
            }
        });
    };

    // ──────────────────────────────────────────────
    // 7. CSS
    // ──────────────────────────────────────────────
    var createCss = function () {
        var css = [
            // JetBrains Mono only — Sora/Inter are gone from the title page
            // (General Sans is injected separately from the project's fonts/
            // assets, with a Fontshare CDN fallback).
            '@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500&display=swap");',

            '#playcanvas-wrapper, #playcanvas-wrapper * {',
            '    box-sizing: border-box;',
            '    margin: 0;',
            '    padding: 0;',
            '  line-height: normal !important;',
            '}',

            /* DISABLE TEXT SELECTION */
            '#custom-splash-wrapper, #custom-splash-wrapper * {',
            '    user-select: none;',
            '    -webkit-user-select: none;',
            '    -ms-user-select: none;',
            '}',

            /* FIXED FULLSCREEN ROOT — black base so we start fully black */
            '#custom-splash-wrapper {',
            '    position: fixed;',
            '    top: 0;',
            '    left: 0;',
            '    width: 100vw;',
            '    height: 100svh;',
            '    background-color: #000000;',
            '    z-index: 1000;',
            '    overflow: hidden;',
            '}',

            /* Once the bands start opening, the root itself must be see-through
               (the bands carry all the black from here on). */
            '#custom-splash-wrapper.bands-open {',
            '    background-color: transparent;',
            '}',

            /* Black 50% layer in front of the scene — fades on CTA click */
            '#custom-splash-wrapper::after {',
            '    content: "";',
            '    position: absolute;',
            '    top: 0;',
            '    left: 0;',
            '    width: 100%;',
            '    height: 100%;',
            '    background: rgba(0,0,0,0.5);',
            '    pointer-events: none;',
            '    opacity: 1;',
            '    z-index: 1;',
            '    transition: opacity 1.2s cubic-bezier(0.4, 0, 0.2, 1);',
            '}',

            /* LOGO — z-index 4: ABOVE the bands. It draws on the black screen
               during loading and stays on top while the bands open under it. */
            '#custom-splash-logo {',
            '    position: absolute;',
            '    top: 50%;',
            '    left: 50%;',
            '    transform: translate(-50%, -50%) translateZ(0);',
            '    width: ' + SPLASH_LOGO_WIDTH + 'px;',
            '    max-width: 85vw;',
            '    z-index: 4;',
            // Promote to its own compositing layer and isolate its rendering so
            // the heavy PlayCanvas preload work doesn't drag the reveal into the
            // same paint/layout pass (mitigates micro-freezes during loading).
            '    will-change: transform;',
            '    contain: layout paint style;',
            '    backface-visibility: hidden;',
            '}',

            // Mobile: shrink the loading (centred) logo by 15%. Only affects the
            // pre-dock loading state — once it docks to the header, dockLogo sets an
            // inline width from OverlayManager's metrics, which overrides this.
            '@media (max-width: 768px) {',
            '    #custom-splash-logo {',
            '        width: ' + Math.round(SPLASH_LOGO_WIDTH * 0.85) + 'px;',   // 430 → 366
            '        max-width: 72.25vw;',   // 85vw × 0.85
            '    }',
            '}',

            '#custom-splash-logo svg {', 
            '    display: block;',
            '    width: 100%;',
            '    height: auto;',
            '    transform: translateZ(0);',
            '}',

            /* BLACK BANDS — z-index 3: they ARE the black screen. Closed for the
               whole load; opened AFTER loading with a composited translateY. */
            '#custom-band-top, #custom-band-bottom {',
            '    position: absolute;',
            '    left: 0;',
            '    width: 100%;',
            '    height: 50%;',
            '    background-color: #000000;',
            '    z-index: 3;',
            '    pointer-events: none;',
            '    will-change: transform;',
            '    backface-visibility: hidden;',
            '    transform: translateY(0);',   // closed: both bands cover their half
            '}',
            '#custom-band-top { top: 0; }',
            '#custom-band-bottom { bottom: 0; }',

            /* The one open move — loading is done, the thread is free. */
            '#custom-splash-wrapper.bands-open #custom-band-top {',
            '    transition: transform ' + BAND_OPEN_MS + 'ms cubic-bezier(0.65, 0, 0.35, 1);',
            '    transform: translateY(-100%);',
            '}',
            '#custom-splash-wrapper.bands-open #custom-band-bottom {',
            '    transition: transform ' + BAND_OPEN_MS + 'ms cubic-bezier(0.65, 0, 0.35, 1);',
            '    transform: translateY(100%);',
            '}',

            /* PROGRESS READOUT — 20px below the logo (top set by JS).
               A centred flex row: [ masked digits column ][ static "%" ].
               No opacity transition on the row itself: at 100% it just
               disappears. */
            '#custom-progress-text {',
            '    position: absolute;',
            '    top: 60%;',
            '    left: 50%;',
            '    transform: translateX(-50%);',
            '    display: flex;',
            '    align-items: center;',
            '    justify-content: center;',
            '    color: #ffffff;',
            '    font-family: "JetBrains Mono", monospace;',
            '    font-size: 0.75rem;',
            '    letter-spacing: 0.15em;',
            '    opacity: 0.7;',
            '    z-index: 4;',
            '}',

            /* One digit column: a fixed-height, ONE-DIGIT-WIDE window. There are
               three (hundreds/tens/units) so total width is constant — the "%"
               never gets pushed when the count reaches 100 or crosses 9→10.
               overflow:hidden masks whatever sits above/below the single visible
               line so each digit reads as a clean reel. */
            '#custom-progress-text .roll-col {',
            '    position: relative;',
            '    display: inline-block;',
            '    width: 1ch;',
            '    height: 1.35em;',
            '    overflow: hidden;',
            // Zeroed so 1ch matches the rendered monospace digit exactly.
            '    letter-spacing: 0;',
            '}',

            /* The static "%" — hugs the digits, never animates. */
            '#custom-progress-text .pct-sign {',
            '    display: inline-block;',
            '    line-height: 1.35em;',
            '    margin-left: 0.15em;',
            '}',

            /* The two number layers of the reel (the resting number + the one
               stepping in). They fill the constant-width column and are
               positioned by JS each frame (see renderProgress) — NO CSS
               transition, so a step can never be interrupted by the next. Centred
               so the number stays visually balanced under the logo. */
            '#custom-progress-text .roll-num {',
            '    position: absolute;',
            '    left: 0;',
            '    right: 0;',
            '    top: 0;',
            '    height: 1.35em;',
            '    line-height: 1.35em;',
            '    text-align: center;',
            '    will-change: transform;',
            '}',

            /* ───── TITLE / CTA (revealed after the bands open) ───── */
            /* SAFARI FIRST-LOAD GUARD — WebKit runs CSS transitions on the very
               first style resolution, so on an uncached first load the title
               group + headphones notice (both z-index 4, above the still-closed
               bands) flashed in at opacity 1 before easing to their real 0 —
               "the bands are behind the title page, then it disappears". Killing
               every splash transition until the first frame has painted lets the
               opacity:0 state show with no animation. The class is dropped via a
               double rAF (showSplash), well before the reveal adds .loaded. */
            '#custom-splash-wrapper.preanim #custom-title-group,',
            '#custom-splash-wrapper.preanim #custom-headphones,',
            '#custom-splash-wrapper.preanim #custom-cta {',
            '    transition: none !important;',
            '}',

            '#custom-title-group {',
            '    position: absolute;',
            '    top: 50%;',
            '    left: 50%;',
            '    transform: translate(-50%, -50%);',
            '    text-align: center;',
            '    opacity: 0;',
            '    transition: opacity 3s ease 0.3s;',
            '    width: 100%;',
            // Responsive side safe-margin: negligible on desktop (wide screen, centred
            // copy), ~24px each side on phones so the title never touches the edges.
            // border-box keeps the padding INSIDE the 100% width (no overflow).
            '    box-sizing: border-box;',
            '    padding: 0 clamp(24px, 6vw, 64px);',
            '    z-index: 4;',
            '}',

            /* General Sans Light 96px uppercase (clamped for small screens) */
            '#custom-title {',
            '    font-family: "General Sans", sans-serif;',
            '    font-weight: 300;',
            '    font-size: clamp(44px, 8.9vw, 96px);',
            '    line-height: 1.04 !important;',
            '    text-transform: uppercase;',
            '    letter-spacing: 0.01em;',
            '    color: #ffffff;',
            '}',

            /* General Sans Regular 16px uppercase */
            '#custom-subtitle {',
            '    font-family: "General Sans", sans-serif;',
            '    font-weight: 400;',
            '    font-size: clamp(11px, 3.2vw, 14px);',
            '    text-transform: uppercase;',
            '    letter-spacing: 0.12em;',
            '    margin-top: 2rem;',
            '    color: #C4C7C7;',
            '    opacity: 0.8;',
            '}',

            /* BOTTOM NOTICE — headphones icon + a subtitle-styled line, pinned to the
               bottom with the same inset the Edolus logo has from the top (40 / 20 px). */
            '#custom-headphones {',
            '    position: absolute;',
            // Full width + flex-centre (NOT left:50% + translateX): a left:50% abs box with
            // right:auto shrink-fits to only ~50% of the viewport, which forced the line to
            // wrap on mobile. Full width lets the text keep its natural single-line width.
            '    left: 0;',
            '    width: 100%;',
            '    box-sizing: border-box;',
            '    bottom: ' + OVERLAY_DEFAULTS.top + 'px;',
            '    display: flex;',
            '    flex-direction: column;',
            '    align-items: center;',
            '    text-align: center;',
            '    opacity: 0;',
            '    transition: opacity 3s ease 0.3s;',   /* matches the title-group reveal */
            '    z-index: 4;',
            '    pointer-events: none;',
            '}',
            '@media (max-width: 768px) {',
            '    #custom-headphones { bottom: ' + OVERLAY_DEFAULTS.topMobile + 'px; }',
            '}',
            '#custom-headphones svg {',
            '    display: block;',
            '    width: 40px;',
            '    height: 40px;',
            '    margin-bottom: 20px;',
            '}',
            // Mobile icon 20% smaller. MUST come AFTER the base svg rule above: media queries
            // add no specificity, so an earlier @media rule would lose the source-order tie to
            // the later 40px base rule (that's why the first attempt showed no change).
            '@media (max-width: 768px) {',
            '    #custom-headphones svg { width: 32px; height: 32px; }',   /* 40 × 0.8 */
            '}',
            '#custom-headphones svg path { fill: #C4C7C7; }',
            /* text: the exact same type as #custom-subtitle (its margin-top is not needed
               here — the icon's 20px margin-bottom sets the gap). */
            '#custom-headphones .hp-text {',
            '    font-family: "General Sans", sans-serif;',
            '    font-weight: 400;',
            '    font-size: clamp(11px, 3.2vw, 14px);',
            '    text-transform: uppercase;',
            '    letter-spacing: 0.12em;',
            '    color: #C4C7C7;',
            '    opacity: 0.8;',
            '    white-space: nowrap;',   /* always one line */
            '}',

            /* Slot: reserves the button's final height + centres it, so the construct-in
               height animation opens from the centre and never reflows the title. */
            '#custom-cta-slot {',
            '    margin-top: 2.75rem;',
            '    display: flex;',
            '    align-items: center;',
            '    justify-content: center;',
            '}',

            '#custom-cta {',
            '    position: relative;',
            '    display: inline-block;',
            '    margin-top: 0;',   // spacing now lives on #custom-cta-slot
            '    padding: 1.2rem 3rem;',
            '    color: #000;',
            '    font-family: "JetBrains Mono", monospace;',
            '    font-weight: 500;',
            '    font-size: 0.75rem;',
            '    letter-spacing: 0.1em;',
            '    background-color: #ffffff;',
            '    opacity: 0;',
            '    cursor: pointer;',
            // Not interactive until revealed — otherwise the invisible button is
            // hoverable/clickable during loading (hand cursor over blank space).
            '    pointer-events: none;',
            '    z-index: 5;',
            '    transition: opacity 1.5s ease, padding 0.3s ease;',
            '}',

            /* Tech corner squares on the CTA.
               Normal state: the squares float OUT around the button
               (4px further horizontally, 2px further vertically than the
               touching/hover position). On hover they slide in to touch the
               button corners. */
            '#custom-cta::before {',
            '    content: "";',
            '    position: absolute;',
            '    top: -6px;',
            '    left: -8px;',
            '    right: -8px;',
            '    bottom: -6px;',
            '    pointer-events: none;',
            '    background-image:',
            '        linear-gradient(#ffffff, #ffffff),',
            '        linear-gradient(#ffffff, #ffffff),',
            '        linear-gradient(#ffffff, #ffffff),',
            '        linear-gradient(#ffffff, #ffffff);',
            '    background-size: 4px 4px;',
            '    background-repeat: no-repeat;',
            '    background-position: top left, top right, bottom left, bottom right;',
            '    transition: top 0.3s ease, bottom 0.3s ease, left 0.3s ease, right 0.3s ease;',
            '}',

            /* Hover: corners slide in to touch the button (the original position) */
            '#custom-cta:hover::before {',
            '    top: -4px;',
            '    left: -4px;',
            '    right: -4px;',
            '    bottom: -4px;',
            '}',

            /* Mobile: tighten the (title/subtitle) → CTA gap, and shrink the CTA ~20%
               (it uses fixed rem, so it didn\'t scale down like the clamp-sized copy). */
            '@media (max-width: 768px) {',
            '    #custom-cta-slot { margin-top: 1.75rem; }',
            '    #custom-cta {',
            '        padding: 0.96rem 2.4rem;',   /* 1.2rem 3rem × 0.8 */
            '        font-size: 0.6rem;',          /* 0.75rem × 0.8 */
            '    }',
            '}',

            /* Hover: the LABEL blinks twice (hard on/off, tech feel), then stays lit.
               Only the text flashes — the white button stays solid. */
            '#cta-label { display: inline-block; }',
            '@keyframes ctaLabelBlink {',
            '    0%   { opacity: 1; }',
            '    25%  { opacity: 0; }',
            '    50%  { opacity: 1; }',
            '    75%  { opacity: 0; }',
            '    100% { opacity: 1; }',
            '}',
            '#custom-cta:hover #cta-label {',
            '    animation: ctaLabelBlink 0.10s steps(1, end);',
            '}',

            '@keyframes ctaBlink {',
            '    0%   { opacity: 1; }',
            '    14%  { opacity: 0; }',
            '    28%  { opacity: 1; }',
            '    42%  { opacity: 0; }',
            '    56%  { opacity: 1; }',
            '    72%  { opacity: 0; }',
            '    100% { opacity: 0; }',
            '}',
            '#custom-cta.blink {',
            '    animation: ctaBlink 0.2s steps(1, end) forwards;',
            '}',

            /* LOADED STATE — bands are gone, reveal the title phase.
               (The logo's move to the header is driven by JS — dockLogo — so it
               can match OverlayManager's exact metrics.) */
            '#custom-splash-wrapper.loaded {',
            '    background-color: transparent;',
            '    pointer-events: none;',
            '}',

            '#custom-splash-wrapper.loaded #custom-title-group {',
            '    opacity: 1;',
            '}',

            '#custom-splash-wrapper.loaded #custom-cta {',
            '    opacity: 1;',
            '    pointer-events: auto;',   // only clickable once revealed
            '}',

            /* CLICKED STATE — the title/subtitle/CTA fade AND blur away
               together (filter on the group blurs its whole subtree), while the
               logo is left untouched and cut in one frame once the overlay logo
               is opaque underneath it. */
            '#custom-splash-wrapper.clicked #custom-title-group {',
            '    opacity: 0;',
            '    filter: blur(14px);',
            '    transition: opacity 1.4s cubic-bezier(0.4, 0, 0.2, 1),',
            '                filter 1.4s cubic-bezier(0.4, 0, 0.2, 1);',
            '}',

            /* Bottom notice fades in with the title and blurs/fades away on CTA click. */
            '#custom-splash-wrapper.loaded #custom-headphones {',
            '    opacity: 1;',
            '}',
            '#custom-splash-wrapper.clicked #custom-headphones {',
            '    opacity: 0;',
            '    filter: blur(14px);',
            '    transition: opacity 1.4s cubic-bezier(0.4, 0, 0.2, 1),',
            '                filter 1.4s cubic-bezier(0.4, 0, 0.2, 1);',
            '}',

            /* On CTA click the black 50% layer fades away */
            '#custom-splash-wrapper.clicked::after {',
            '    opacity: 0;',
            '}'
        ].join('\n');

        var style = document.createElement('style');
        style.appendChild(document.createTextNode(css));
        document.head.appendChild(style);

        // GSAP LOAD
        var script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js';
        document.head.appendChild(script);
    };

    // ──────────────────────────────────────────────
    // INIT
    // ──────────────────────────────────────────────
    createCss();
    showSplash();

    window.addEventListener('resize', onResize);

    rafId = requestAnimationFrame(tick);

    // Warm-up progress feed (SceneManager). Registered at INIT — before the
    // app even starts — so no event can ever be missed. Monotonic: the % must
    // never move backwards.
    window.addEventListener('warmup:progress', function (e) {
        var p = (e && typeof e.detail === 'number') ? e.detail : 0;
        if (p > warmupProgress) warmupProgress = Math.min(1, p);
    });

    // The asset registry is populated by the time preload starts — resolve the
    // General Sans files then ('start' is the idempotent safety net).
    app.once('preload:start', injectGeneralSans);

    app.on('preload:progress', setProgress);

    app.on('preload:end', function () {
        app.off('preload:progress');
    });

    app.on('start', function () {
        injectGeneralSans();
        // App is fully ready. The RAF gate still waits for the logo animation
        // and REVEAL_MIN_MS, so the draw-on is always seen in full.
        realProgress = 1;

        // Hold the reveal until SceneManager's warm-up has walked the whole
        // timeline behind the black screen (compiles every scene's shaders so
        // the first scroll never freezes) AND its settle time has finished
        // easing scene 1 back into place. The % holds at HOLD_BEFORE_OPEN
        // meanwhile.
        //
        // CRITICAL ORDER NOTE: PlayCanvas fires 'start' BEFORE script
        // initialize/postInitialize, so window.__sceneWarmup is NOT set yet
        // when this handler runs — it must never be checked synchronously here
        // (doing so made the gate a no-op and the bands opened mid-walk).
        // Instead: always wait for the warm-up signal; a 1 s probe bails out
        // only if no warm-up ever STARTED (SceneManager missing/stale), and a
        // long stop-loss covers a warm-up that started but never finished.
        var complete = function () { loadingComplete = true; };
        if (window.__warmupDone) { complete(); return; }

        var safety = setTimeout(complete, 25000);   // stop-loss: broken warm-up
        window.addEventListener('warmup:done', function () {
            clearTimeout(safety);
            warmupProgress = 1;
            complete();
        }, { once: true });

        // No warm-up flag one second in = this build has no warm-up → don't
        // hold the black screen for the full stop-loss.
        setTimeout(function () {
            if (!window.__sceneWarmup && !window.__warmupDone) {
                clearTimeout(safety);
                complete();
            }
        }, 1000);
    });
});
