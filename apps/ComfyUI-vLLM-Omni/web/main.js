const MODELS_EXTRA_FIELDS = {
    "ByteDance-Seed/BAGEL-7B-MoT": {
        "modalities": ["text", "image"],
    },
    "Qwen/Qwen2.5-Omni-7B": {
        "modalities": ["text", "audio"],
    },
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": {
        "modalities": ["text", "audio"],
    },
}

import { app } from "../../scripts/app.js";
app.registerExtension({
    name: "vllm.vllm_omni",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log(`Node name ${nodeData.name}`)
        if (!nodeData.name.startsWith("VLLMOmni")) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = async function () {
            const originalReturnValue = onNodeCreated?.apply(this)

            const modelField = this.getField("model")
            if (!modelField) {
                return
            }

            const updateModalitiesField = () => {
                const modelValue = modelField.getValue()
                const extraFields = MODELS_EXTRA_FIELDS[modelValue] || {}
                const modalities = extraFields["modalities"] || ["text"]

                let modalitiesField = this.getField("modalities")
                if (!modalitiesField) {
                    modalitiesField = this.addSelectField("modalities", "Modalities", {
                        options: [],
                        multiple: true,
                    })
                }

                const options = [
                    { value: "text", label: "Text" },
                    { value: "image", label: "Image" },
                    { value: "audio", label: "Audio" },
                ]
                modalitiesField.setOptions(options)
                modalitiesField.setValue(modalities.filter(m => options.some(o => o.value === m)))
            }

            modelField.on("change", updateModalitiesField)
            updateModalitiesField()

            return originalReturnValue
        }
    },
    async setup() {
        console.log("vLLM-Omni Setup complete!")
    },
})