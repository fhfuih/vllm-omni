/**
 * @file This file adds stub code that adds dynamic fields to vLLM-Omni nodes
 * based on widget (in-node form fields) and input (connection link) values and changes.
 * However, this functionality is currently disabled/commented out.
 * Because it introduces too much complexity,
 * and it may even conflict with the current backend (Python) validation for unknown reasons (pending ConfyUI upstream fixes).
 */

import { app } from "../../scripts/app.js";
import { addMultilineWidget } from "./utils.js";
app.registerExtension({
    name: "vllm.vllm_omni",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData.name.startsWith("VLLMOmni")) {
            return
        }

        let eventHandlers = eventHandlerRegistry[nodeData.name] || {}

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = async function () {
            // console.debug('this (node)', this)
            const originalReturnValue = originalOnNodeCreated?.apply(this)

            this.widgets.forEach(field => {
                // console.debug('field', field.name, field)
                const fieldCallback = eventHandlers[field.name]
                if (!fieldCallback) return
                field.callback = fieldCallback.bind(field)
                fieldCallback.call(field, field.value, app.canvas, this, null, null)
            })

            this.setDirtyCanvas(true, false)
            return originalReturnValue
        }
    },
    async setup() {
        console.info("vLLM-Omni Setup complete!")
    },
})

const eventHandlerRegistry = {
    VLLMOmniGenerateImage: {},
    VLLMOmniComprehension: {},
    VLLMOmniGenerateVideo: {},
    VLLMOmniTTS: {
        // model: addQwenTTSFieldsOnModelChange
    },
    VLLMOmniVoiceClone: {
        // model: addQwenTTSFieldsOnModelChange
    }
}

// function addQwenTTSFieldsOnModelChange(value, _canvas, node, _pos, _event) {
//     const isQwenTTS = value.toLowerCase().includes("qwen3-tts")
//     console.debug('this field', this)


//     const lastRequiredFieldIdxInNode = node.widgets.findIndex(w => w.name == 'speed')
//     if (lastRequiredFieldIdxInNode === -1) {
//         console.warn("Could not find 'speed' field in node widgets.")
//         return
//     }
//     node.widgets.slice(lastRequiredFieldIdxInNode + 1).forEach(w => {
//         node.removeWidget(w)
//     })
//     let specialFields = {}
//     if (isQwenTTS) {
//         specialFields = {
//             ...specialFields,
//             task_type: {
//                 type: "combo",
//                 options: {
//                     values: ["CustomVoice", "VoiceDesign", "Base"]
//                 },
//                 value: "CustomVoice",
//                 callback: null
//             },
//             language: {
//                 type: "combo",
//                 options: {
//                     values: ["Auto", "Chinese", "English", "Japanese", "Korean"]
//                 },
//                 value: "Auto",
//                 callback: null
//             },
//             instructions: {
//                 type: "string",
//                 options: { multiline: true },
//                 value: "",
//                 callback: null
//             },
//             max_new_tokens: {
//                 type: "int",
//                 options: { min: 1 },
//                 value: 2048,
//                 callback: null
//             }
//         }
//     }
//     Object.entries(specialFields).forEach(([fieldName, fieldDef]) => {
//         const multiline = fieldDef.options?.multiline
//         if (multiline) {
//             addMultilineWidget(node, fieldName, {
//                 defaultVal: fieldDef.value,
//             })
//         } else {
//             node.addWidget(
//                 fieldDef.type,
//                 fieldName,
//                 fieldDef.value,
//                 fieldDef.callback || (() => { }),
//                 fieldDef.options || {},
//             )
//         }
//     })
// }
