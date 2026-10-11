// Arrow-key paging for the marker-snapped window selection.
//
// Left/Right shift the window that is currently hovered (tracked via the
// nearest ancestor carrying data-selection-scope) to the previous/next
// marker pair. The actual window math lives in Python
// (callbacks.page_selection_by_arrow_key / _page_selection); this file only
// figures out which graph is hovered and forwards a (scope, direction) event
// into the "arrow-key-event" dcc.Store.
(function attachKeyboardNav() {
    if (window.__hmaKeyNavAttached) {
        return;
    }
    window.__hmaKeyNavAttached = true;

    var hoveredScope = null;
    var seq = 0;

    function scopeOf(target) {
        var el = target && target.closest ? target.closest("[data-selection-scope]") : null;
        return el ? el.getAttribute("data-selection-scope") : null;
    }

    document.addEventListener(
        "mouseover",
        function (event) {
            var scope = scopeOf(event.target);
            if (scope) {
                hoveredScope = scope;
            }
        },
        true
    );

    document.addEventListener(
        "mouseout",
        function (event) {
            var scope = scopeOf(event.target);
            if (scope && scope === hoveredScope) {
                // Only clear if the pointer actually left the scoped element,
                // not just moved between its children.
                var related = event.relatedTarget;
                if (!related || scopeOf(related) !== scope) {
                    hoveredScope = null;
                }
            }
        },
        true
    );

    document.addEventListener("keydown", function (event) {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
            return;
        }
        var target = event.target;
        var tag = target && target.tagName;
        if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || (target && target.isContentEditable)) {
            return;
        }
        if (!hoveredScope) {
            return;
        }
        if (!window.dash_clientside || !window.dash_clientside.set_props) {
            return;
        }
        event.preventDefault();
        seq += 1;
        window.dash_clientside.set_props("arrow-key-event", {
            data: {
                scope: hoveredScope,
                direction: event.key === "ArrowRight" ? "next" : "prev",
                seq: seq,
            },
        });
    });
})();
