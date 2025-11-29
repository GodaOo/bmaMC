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

## Known Issues

1. **Model size limits**

   Minecraft requires each cube’s `from` / `to` coordinates to stay within the range **-16 to 32** on every axis — this includes all positions reached during animation.  
   If any element goes outside this range at any frame, the model will not render.

   **Workaround: scale down in Blockbench, then scale back up in bmaMC**

   1. In Blockbench, open **File → Project…** and set **Default UV mode** to **Per-Face UV**.  
   2. Use **Transform → Scale…**, with the **pivot at (0, 0, 0)**, to scale the whole model down.  
   3. The recommended minimum scale factor is **0.25**. Going smaller than 0.25 cannot be fully restored later, because Minecraft only supports up to **4×** scale back up.  
   4. Ideally, adjust the model size **before** creating animations.  
      - If you already have animations, you must manually scale all animated **Position** values:  
        each keyframe position should be multiplied by the same scale factor you used on the model (for example, original position `10` with scale `0.25` becomes `2.5`).
   5. In **bmaMC**, set the **Scale** field to the **inverse** of the Blockbench scale you used.  
      - Example: if you scaled the model to **0.5** in Blockbench, then set **Scale = 2** in bmaMC (`1 / 0.5 = 2`) to restore its in-game size.

2. **Single-sided geometry**

   Some models (for example, `evoker_fangs`) appear to show “inside” faces in Blockbench.  
   However, due to Minecraft’s rendering rules, faces are only rendered on the **outer side**; the back side (inside) is not drawn.  

   This means **single-sided** or thin, hollow models that rely on visible interior faces will not display correctly in-game.  
   When designing models for use with bmaMC, avoid relying on geometry where you need to see the inside of a single face or thin shell.
   ![face_issue](https://github.com/user-attachments/assets/d035965d-bbac-43ea-8a1a-ef5eef150346)

## Acknowledgments

Special thanks to **ChatGPT** for development assistance and idea refinement,  
and to **Blockbench** for providing such a powerful and user-friendly modeling tool  
that made this project possible.
