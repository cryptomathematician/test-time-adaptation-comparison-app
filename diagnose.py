import sys, os

cwd = os.getcwd()
print("CWD:", cwd)

# LAN
lan = os.path.join(cwd, "models", "LAN")
print("\n--- LAN ---")
print("Path exists:", os.path.exists(lan))
print("model.py:", os.path.exists(os.path.join(lan, "model.py")))
print("Restormer dir:", os.path.exists(os.path.join(lan, "Restormer")))
print("Files in LAN:", os.listdir(lan))

# TAO
tao = os.path.join(cwd, "models", "2024-ICML-TAO")
print("\n--- TAO ---")
print("Path exists:", os.path.exists(tao))
gd = os.path.join(tao, "gen_dif_pri")
print("gen_dif_pri exists:", os.path.exists(gd))
print("gen_dif_pri/__init__.py:", os.path.exists(os.path.join(gd, "__init__.py")))
gs = os.path.join(gd, "scripts")
print("scripts exists:", os.path.exists(gs))
print("scripts/__init__.py:", os.path.exists(os.path.join(gs, "__init__.py")))
gd_s = os.path.join(gs, "guided_diffusion")
print("guided_diffusion exists:", os.path.exists(gd_s))
print("guided_diffusion/__init__.py:", os.path.exists(os.path.join(gd_s, "__init__.py")))
print("script_util_x0.py:", os.path.exists(os.path.join(gd_s, "script_util_x0.py")))

# TTAD
ttad = os.path.join(cwd, "models", "TTAD")
print("\n--- TTAD ---")
print("Path exists:", os.path.exists(ttad))
print("arch_unet.py:", os.path.exists(os.path.join(ttad, "arch_unet.py")))
print("epoch_model_100.pth:", os.path.exists(os.path.join(ttad, "epoch_model_100.pth")))
print("Files in TTAD:", os.listdir(ttad))

# Try importing
print("\n--- Trying imports ---")
sys.path.insert(0, lan)
sys.path.insert(0, tao)
sys.path.insert(0, ttad)

# Import in the order that would work
try:
    import arch_unet
    print("arch_unet: OK")
except Exception as e:
    print("arch_unet:", e)

try:
    from gen_dif_pri.scripts.guided_diffusion import script_util_x0
    print("script_util_x0: OK")
except Exception as e:
    print("script_util_x0:", e)

try:
    # Need to cd into lan dir first for model.py's relative import
    old = os.getcwd()
    os.chdir(lan)
    import model
    print("model: OK")
    os.chdir(old)
except Exception as e:
    os.chdir(cwd)
    print("model:", e)