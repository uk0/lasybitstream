import json, os, numpy as np, torch
from safetensors import safe_open
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision("highest")
MODEL="/model"; OUT="/work/test"; os.makedirs(OUT,exist_ok=True)
from transformers import AutoConfig
cfg=AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
vcfg=cfg.vision_config
print("vcfg depth",vcfg.depth,"hidden",vcfg.hidden_size,"heads",vcfg.num_heads,"merge",vcfg.spatial_merge_size)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
vm=Qwen3_5VisionModel(vcfg).to(torch.float32).eval()
f=safe_open(MODEL+"/model.safetensors",framework="pt")
sd={}
for k in f.keys():
    if k.startswith("model.visual."): sd[k[len("model.visual."):]]=f.get_tensor(k).float()
miss,unexp=vm.load_state_dict(sd,strict=False)
print("loaded visual: missing",len(miss),"unexpected",len(unexp), "params",len(sd))
# test image via processor
from PIL import Image
try:
    from transformers import AutoProcessor
    proc=AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    img=Image.new("RGB",(112,140),(30,90,160))
    for x in range(112):
        for y in range(140): img.putpixel((x,y),((x*2)%256,(y*3)%256,((x+y))%256))
    msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"describe"}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=proc(text=[text],images=[img],return_tensors="pt")
    pv=inputs["pixel_values"].float(); grid=inputs["image_grid_thw"]
    print("pixel_values",tuple(pv.shape),"grid_thw",grid.tolist(),"input_ids",inputs["input_ids"].shape)
    caps={}
    def pre_hook(m,a): caps["in"]=a[0].detach().float().cpu().numpy()  # block0 input = patch+pos
    def b0_hook(m,a,o): caps["b0"]=o.detach().float().cpu().numpy()
    def n2_hook(m,a): caps["postattn"]=a[0].detach().float().cpu().numpy()  # norm2 input = h+attn
    vm.blocks[0].register_forward_pre_hook(pre_hook)
    vm.blocks[0].register_forward_hook(b0_hook)
    vm.blocks[0].norm2.register_forward_pre_hook(n2_hook)
    out=vm(pv,grid)
    emb=out.pooler_output.float()
    caps["in"].astype(np.float32).tofile(OUT+"/vis_h_in.f32")
    caps["b0"].astype(np.float32).tofile(OUT+"/vis_h_b0.f32")
    caps["postattn"].astype(np.float32).tofile(OUT+"/vis_h_postattn.f32")
    out.last_hidden_state.float().cpu().numpy().astype(np.float32).tofile(OUT+"/vis_h_pre.f32")
    print("dumped intermediates: in",caps["in"].shape,"b0",caps["b0"].shape,"pre",tuple(out.last_hidden_state.shape))
    print("merged emb",tuple(emb.shape),"mean",float(emb.mean()),"std",float(emb.std()))
    pv.cpu().numpy().astype(np.float32).tofile(OUT+"/vis_pixels.f32")
    grid.cpu().numpy().astype(np.int32).tofile(OUT+"/vis_grid.i32")
    emb.cpu().numpy().astype(np.float32).tofile(OUT+"/vis_emb.f32")
    inputs["input_ids"][0].cpu().numpy().astype(np.int32).tofile(OUT+"/vis_input_ids.i32")
    json.dump({"pv_shape":list(pv.shape),"grid":grid.tolist(),"emb_shape":list(emb.shape),
               "n_ids":int(inputs["input_ids"].shape[1])}, open(OUT+"/vis_meta.json","w"))
    print("WROTE vision reference")
except Exception as e:
    import traceback; traceback.print_exc()
