# bmaMC (Block Model Animation Minecraft)

<img width="64" height="64" alt="icon" src="https://github.com/user-attachments/assets/1e859feb-af52-42f2-acff-b0753b304ad6" />

Convert **Blockbench Bedrock entity animations** into **Minecraft Java Edition resource-pack models** with **multi-axis rotation** (25w46a+).

## UI

<img height="300" alt="image" src="https://github.com/user-attachments/assets/c06bf067-8480-48f6-8558-84fa85ff7dd7" />

## Notes

- **Minecraft 25w46a+ only** (uses per-element XYZ rotation).
- Animation import supports **translation** and **rotation**; **scale is read but ignored** to avoid distortion.

## Usage

1. Create your **Bedrock Entity** model and animations in **Blockbench**. Save project as `.bbmodel` file.
2. Open the bmaMC GUI:
   - Run from terminal: `python gui.py`
3. Import one or more `.bbmodel` files into **bmaMC** (multi-file import supported).
4. On the right panel, set:
   - **Output folder**
   - **Description** (for `pack.mcmeta`)
   - **Pack format** — see the list here: https://minecraft.wiki/w/Pack_format#List_of_resource_pack_formats
   - **FPS** (recommended **20**; no in-between frame interpolation, so lower FPS will look choppy)
5. Click **Generate** or **Clear Folder and Generate**  
   - *Generate*: builds into the selected output folder  
   - *Clear Folder and Generate*: empties that folder first, then builds
6. Load the output as a **resource pack** in Minecraft.

**Example (give item using a dynamic model):**  
If the model name is `model` and the animation name is `walk`, you can get an apple using the animated model with:
```mcfunction
/give @a minecraft:apple[minecraft:item_model="model/walk"]
```
![example](https://github.com/user-attachments/assets/346b3cc2-6193-4781-ab94-fc917fd062b6)

## Acknowledgments

Special thanks to **ChatGPT** for development assistance and idea refinement,  
and to **Blockbench** for providing such a powerful and user-friendly modeling tool  
that made this project possible.
