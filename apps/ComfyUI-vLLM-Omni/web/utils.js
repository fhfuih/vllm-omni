/**
 * @link https://github.com/Comfy-Org/ComfyUI_frontend/blob/507500a9d76f3ce0e9bc4738c796ec632701ca5b/src/renderer/extensions/vueNodes/widgets/composables/useStringWidget.ts
 */

export function addMultilineWidget(
    node,
    name,
    opts
) {
    const inputEl = document.createElement('textarea')
    inputEl.className = 'comfy-multiline-input'
    inputEl.value = opts.defaultVal
    inputEl.placeholder = opts.placeholder || name

    const widget = node.addDOMWidget(name, 'customtext', inputEl, {
        getValue() {
            return inputEl.value
        },
        setValue(v) {
            inputEl.value = v
        }
    })

    widget.inputEl = inputEl
    widget.options.minNodeSize = [400, 200]

    inputEl.addEventListener('input', () => {
        widget.callback?.(widget.value)
    })

    // Allow middle mouse button panning
    inputEl.addEventListener('pointerdown', (event) => {
        if (event.button === 1) {
            app.canvas.processMouseDown(event)
        }
    })

    inputEl.addEventListener('pointermove', (event) => {
        if ((event.buttons & 4) === 4) {
            app.canvas.processMouseMove(event)
        }
    })

    inputEl.addEventListener('pointerup', (event) => {
        if (event.button === 1) {
            app.canvas.processMouseUp(event)
        }
    })

    inputEl.addEventListener('wheel', (event) => {
        const gesturesEnabled = useSettingStore().get(
            'LiteGraph.Pointer.TrackpadGestures'
        )
        const deltaX = event.deltaX
        const deltaY = event.deltaY

        const canScrollY = inputEl.scrollHeight > inputEl.clientHeight
        const isHorizontal = Math.abs(deltaX) > Math.abs(deltaY)

        // Prevent pinch zoom from zooming the page
        if (event.ctrlKey) {
            event.preventDefault()
            event.stopPropagation()
            app.canvas.processMouseWheel(event)
            return
        }

        // Detect if this is likely a trackpad gesture vs mouse wheel
        // Trackpads usually have deltaX or smaller deltaY values (< TRACKPAD_DETECTION_THRESHOLD)
        // Mouse wheels typically have larger discrete deltaY values (>= TRACKPAD_DETECTION_THRESHOLD)
        const isLikelyTrackpad =
            Math.abs(deltaX) > 0 || Math.abs(deltaY) < TRACKPAD_DETECTION_THRESHOLD

        // Trackpad gestures: when enabled, trackpad panning goes to canvas
        if (gesturesEnabled && isLikelyTrackpad) {
            event.preventDefault()
            event.stopPropagation()
            app.canvas.processMouseWheel(event)
            return
        }

        // When gestures disabled: horizontal always goes to canvas (no horizontal scroll in textarea)
        if (isHorizontal) {
            event.preventDefault()
            event.stopPropagation()
            app.canvas.processMouseWheel(event)
            return
        }

        // Vertical scrolling when gestures disabled: let textarea scroll if scrollable
        if (canScrollY) {
            event.stopPropagation()
            return
        }

        // If textarea can't scroll vertically, pass to canvas
        event.preventDefault()
        app.canvas.processMouseWheel(event)
    })

    return widget
}
