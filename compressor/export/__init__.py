"""export layer: organized by deployment target.

Includes:
- onnx_export → QDQ ONNX(CNN/ViT) + DeploymentManifest
- rlym_builder → ONNX conversion to .rlym(NNE builder, callback to nnexec)
- hf_export → LLM/VLM compressed model
- vllm_validate → vllm-csk deployment condition check
"""
