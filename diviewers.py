import tkinter as tk
import tkinter.ttk as ttk
from tkinter import scrolledtext
import json
import os
import sys
import subprocess
from tkinter import filedialog
from subprocess import PIPE
import threading
from diffusers.models import AutoencoderKL
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline, StableDiffusionControlNetPipeline
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
)
from LoadLora import load_safetensors_lora  # @UnresolvedImport
import datetime
from time import sleep
import random
import copy
import glob
from PIL import Image, ImageTk,ExifTags
import io
from contextlib import redirect_stderr, redirect_stdout
import gc
from memory_profiler import profile


# import playsound

class GUI():
    def __init__(self, root):
        root.title("Switching Function")
        root.geometry("+0+0")
        
        button_generate_image = tk.Button(
            root,
            text = "Generating Image",
            bg = "white",
            fg = "black",
            width = 15,
            command = self.start_generating_image
            )
        button_create_lora = tk.Button(
            root,
            text = "Creating LoRA",
            bg = "white",
            fg = "black",
            width = 15,
            command = self.start_creating_lora
            )
        
        button_generate_image.grid(column = 0, row = 0)
        button_create_lora.grid(column = 1, row = 0)
    
    def start_generating_image(self,):
        global gngui
        print("generating_image")
        
        gngui = tk.Toplevel()
        GNIMG(gngui)
    
    def start_creating_lora(self,):
        global crgui
        print("creating_lora")
        
        crgui = tk.Toplevel()
        CRLORA(crgui)
        
class CRLORA():
    def __init__(self, root):
        global frame_menu
        global button_dic
        global combo_dic
        global button_name
        global text_name
        global button_tag
        global combo_tag
        global button_tag_add
        global button_tag_rmv
        global button_prompt
        global combo_prompt_name
        global combo_prompt_cunt
        global button_prompt_rmv_one
        global button_prompt_rmv_all
        global button_image_to_tag
        
        global text_model_path
        global text_input_path
        global text_output_path
        
        global spin_res_x
        global spin_res_y
        global text_learning_rate
        global spin_unet_rate
        global spin_enco_rate
        global spin_nt_dim
        global spin_nt_alpha
        
        global spin_num_img
        global spin_learning_rate
        global spin_repeats
        global spin_batch
        global spin_epochs
        global spin_goal_rate

        global text_num_img

        global tag_list
        
        tag_list = []
        
        width = [5, 11, 4, 8, 15]
        
        root.title("diviewers")
        root.geometry("+0+0")
            
        frame_menu = tk.Frame(root)
                
        
        frame_tag = tk.Frame(frame_menu)
        button_tag = tk.Button(
            frame_tag,
            text = "Tag",
            bg = "white",
            fg = "black",
            width = width[1],
            relief = tk.RAISED,
            )
        combo_tag = ttk.Combobox(
            frame_tag,
            )
        button_tag_add = tk.Button(
            frame_tag,
            text = "＋",
            bg = "white",
            fg = "black",
            width = width[1],
            relief = tk.RAISED,
            command = lambda: self.button_add_prompt(),
            )
        button_tag_rmv = tk.Button(
            frame_tag,
            text = "ー",
            bg = "white",
            fg = "black",
            width = width[1],
            relief = tk.RAISED,
            command = lambda: self.button_tag_to_prompt(),
            )
        
        frame_prompt = tk.Frame(frame_menu)
        button_prompt = tk.Button(
            frame_prompt,
            text = "Prompt",
            bg = "white",
            fg = "black",
            width = width[1],
            relief = tk.RAISED,
            )
        combo_prompt_name = ttk.Combobox(
            frame_prompt,
            )
        combo_prompt_name.bind("<<ComboboxSelected>>",self.combo_prompt_name_set)
        combo_prompt_cunt = ttk.Combobox(
            frame_prompt,
            width = 3,
            )
        button_prompt_rmv_one = tk.Button(
            frame_prompt,
            text = "Remove one",
            bg = "white",
            fg = "black",
            # width = 10,
            relief = tk.RAISED,
            command = lambda: self.button_prompt_to_tag_one(),
            )
        button_prompt_rmv_all = tk.Button(
            frame_prompt,
            text = "Remove all",
            bg = "white",
            fg = "black",
            # width = 10,
            relief = tk.RAISED,
            command = lambda: self.button_prompt_to_tag_all(),
            )
        
        button_edit_tag = tk.Button(
            frame_menu,
            text = "Edit tags",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            command = self.edit_tag,
            )

        button_img_to_tag = tk.Button(
            frame_menu,
            text = "Image to Tag",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            # width = width[1],
            command = self.image_to_tag, 
            )
        
        button_tag_to_lora = tk.Button(
            frame_menu,
            text = "Tag to LoRA",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            # width = width[1],
            command = self.tag_to_lora,
            )   
    
        frame_model = tk.Frame(frame_menu)
        button_model_path = tk.Button(
            frame_model,
            text = "Model path",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            width = width[1],
            command = lambda: self.ref_path("model"),
            )
        text_model_path = tk.Entry(
            frame_model,
            width = width[4],
            )
        
        frame_input = tk.Frame(frame_menu)
        button_input_path = tk.Button(
            frame_input,
            text = "Input path",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            width = width[1],
            command = lambda: self.ref_path("input"),
            )
        text_input_path = tk.Entry(
            frame_input,
            width = width[4],
            )
        button_num_img = tk.Button(
            frame_input,
            text = "Number of images",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            command = lambda: self.count_images(),
            )
        text_num_img = tk.Entry(
            frame_input,
            width = width[0]
            )
        
        frame_output = tk.Frame(frame_menu)
        button_output_path = tk.Button(
            frame_output,
            text = "Output path",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            width = width[1],
            command = lambda: self.ref_path("output"),
            )
        text_output_path = tk.Entry(
            frame_output,
            width = width[4],
            )
        text_output_path.insert(0,r"E:/lora")
        
        frame_rate = tk.Frame(frame_menu)
        button_learning_rate = tk.Button(
            frame_rate,
            text = "Learning rate",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            width = width[1],
            command = self.auto_set_rate,
            )
        spin_learning_rate = ttk.Spinbox(
            frame_rate,
            from_ = 0,
            to = 0.01,
            increment = 0.00001,
            width = width[3],
            )
        spin_learning_rate.set(0.0001)
        button_unet = tk.Button(
            frame_rate,
            text = "Unet",   
            command = self.copy_unet,     
            )
        spin_unet_rate = ttk.Spinbox(
            frame_rate,
            from_ = 0,
            to = 0.01,
            increment = 0.00001,
            width = width[3],
            )
        spin_unet_rate.set(0.0001)
        button_enco = tk.Button(
            frame_rate,
            text = "Encorder",        
            command = self.copy_enco,
            )
        spin_enco_rate = ttk.Spinbox(
            frame_rate,
            from_ = 0,
            to = 0.01,
            increment = 0.00001,
            width = width[3],
            )
        spin_enco_rate.set(0.0001)
        label_goal_rate = tk.Label(
            frame_rate,
            text = "goal",
            )
        spin_goal_rate = ttk.Spinbox(
            frame_rate,
            from_ = 0.01,
            to = 1.00,
            increment = 0.01,
            width = width[0]
            )
        spin_goal_rate.set(0.95)
        
        
        frame_num = tk.Frame(frame_menu)
        button_repeats = tk.Button(
            frame_num,
            text = "Repeats",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            width = width[1],
            command = self.tag_to_lora,
            )
        spin_repeats = ttk.Spinbox(
            frame_num,
            from_ = 1,
            to = 50,
            increment = 1,
            width = width[0],
            )
        spin_repeats.set(10)
        button_batch = tk.Button(
            frame_num,
            text = "Batch size",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            width = width[1],
            command = self.tag_to_lora,
            )
        spin_batch = ttk.Spinbox(
            frame_num,
            from_ = 1,
            to = 5,
            increment = 1,
            width = width[0],
            )
        spin_batch.set(2)
        button_epochs = tk.Button(
            frame_num,
            text = "Epochs",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            width = width[1],
            command = self.tag_to_lora,
            )
        spin_epochs = ttk.Spinbox(
            frame_num,
            from_ = 1,
            to = 50,
            increment = 1,
            width = width[0],
            )
        spin_epochs.set(4)
        
        frame_res = tk.Frame(frame_menu)
        button_res = tk.Button(
            frame_res,
            text = "Resolution",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            width = width[1],
            command = self.tag_to_lora,
            )
        label_res_x = tk.Label(
            frame_res,
            text = "X",        
            width = width[2],
            )
        spin_res_x = ttk.Spinbox(
            frame_res,
            from_ = 64,
            to = 1024,
            increment = 64,
            width = width[0],
            )
        spin_res_x.set(640)
        label_res_y = tk.Label(
            frame_res,
            text = "Y",    
            width = width[2],    
            )
        spin_res_y = ttk.Spinbox(
            frame_res,
            from_ = 64,
            to = 1024,
            increment = 64,
            width = width[0],
            )
        spin_res_y.set(512)
        
        frame_nt = tk.Frame(frame_menu)
        button_nt = tk.Button(
            frame_nt,
            text = "Network",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            width = width[1],
            command = self.tag_to_lora,
            )
        label_dim = tk.Label(
            frame_nt,
            text = "Dim",    
            width = width[2],    
            )
        spin_nt_dim = ttk.Spinbox(
            frame_nt,
            from_ = 1,
            to = 256,
            increment = 1,
            width = width[0],
            )
        spin_nt_dim.set(4)
        label_alpha = tk.Label(
            frame_nt,
            text = "Alpha",
            width = width[2],    
            )
        spin_nt_alpha = ttk.Spinbox(
            frame_nt,
            from_ = 1,
            to = 256,
            increment = 1,
            width = width[0],
            )
        spin_nt_alpha.set(1)
        
        frame_menu.grid(column = 0, row = 0)
        
        frame_input.grid(column = 0, row = 0, sticky = tk.NSEW)
        button_input_path.grid(column = 0, row = 0, sticky = tk.NSEW)
        text_input_path.grid(column = 1, row = 0, sticky = tk.NSEW)
        button_num_img.grid(column = 2, row = 0, sticky = tk.NSEW)
        text_num_img.grid(column = 3, row = 0, sticky = tk.NSEW)
        
        button_img_to_tag.grid(column = 0, row = 1, sticky = tk.NSEW)
        
        frame_tag.grid(column = 0, row = 2, sticky = tk.NSEW)
        button_tag.grid(column = 0, row = 0, sticky = tk.NSEW)
        combo_tag.grid(column = 1, row = 0, sticky = tk.NSEW)
        button_tag_add.grid(column = 2, row = 0, sticky = tk.NSEW)
        button_tag_rmv.grid(column = 3, row = 0, sticky = tk.NSEW)
    
        frame_prompt.grid(column = 0, row = 3, sticky = tk.NSEW)
        button_prompt.grid(column = 0, row = 0, sticky = tk.NSEW)
        combo_prompt_name.grid(column = 1, row = 0, sticky = tk.NSEW)
        combo_prompt_cunt.grid(column = 2, row = 0, sticky = tk.NSEW)
        button_prompt_rmv_one.grid(column = 3, row = 0, sticky = tk.NSEW)
        button_prompt_rmv_all.grid(column = 4, row = 0, sticky = tk.NSEW)
        
        button_edit_tag.grid(column = 0, row = 4, sticky = tk.NSEW)
        
        frame_model.grid(column = 0, row = 5, sticky = tk.NSEW)
        button_model_path.grid(column = 0, row = 0, sticky = tk.NSEW)
        text_model_path.grid(column = 1, row = 0, sticky = tk.NSEW)
        
        frame_output.grid(column = 0, row = 6, sticky = tk.NSEW)
        button_output_path.grid(column = 0, row = 0, sticky = tk.NSEW)
        text_output_path.grid(column = 1, row = 0, sticky = tk.NSEW)
        
        frame_rate.grid(column = 0, row = 7, sticky = tk.NSEW)
        button_learning_rate.grid(column = 0, row = 0, sticky = tk.NSEW)
        spin_learning_rate.grid(column = 1, row = 0, sticky = tk.NSEW)
        button_unet.grid(column = 2, row = 0, sticky = tk.NSEW)
        spin_unet_rate.grid(column = 3, row = 0, sticky = tk.NSEW)
        button_enco.grid(column = 4, row = 0, sticky = tk.NSEW)
        spin_enco_rate.grid(column = 5, row = 0, sticky = tk.NSEW)
        label_goal_rate.grid(column = 6, row = 0, sticky = tk.NSEW)
        spin_goal_rate.grid(column = 7, row = 0, sticky = tk.NSEW)
        
        frame_num.grid(column = 0, row = 8, sticky = tk.NSEW)
        button_repeats.grid(column = 0, row = 0, sticky = tk.NSEW)
        spin_repeats.grid(column = 1, row = 0, sticky = tk.NSEW)
        button_batch.grid(column = 2, row = 0, sticky = tk.NSEW)
        spin_batch.grid(column = 3, row = 0, sticky = tk.NSEW)
        button_epochs.grid(column = 4, row = 0, sticky = tk.NSEW)
        spin_epochs.grid(column = 5, row = 0, sticky = tk.NSEW)
        
        frame_res.grid(column = 0, row = 9, sticky = tk.NSEW)
        button_res.grid(column = 0, row = 0, sticky = tk.NSEW)
        label_res_x.grid(column = 1, row = 0, sticky = tk.NSEW)
        spin_res_x.grid(column = 2, row = 0, sticky = tk.NSEW)
        label_res_y.grid(column = 3, row = 0, sticky = tk.NSEW)
        spin_res_y.grid(column = 4, row = 0, sticky = tk.NSEW)
        
        frame_nt.grid(column = 0, row = 10, sticky = tk.NSEW)
        button_nt.grid(column = 0, row = 0, sticky = tk.NSEW)
        label_dim.grid(column = 1, row = 0, sticky = tk.NSEW)
        spin_nt_dim.grid(column = 2, row = 0, sticky = tk.NSEW)
        label_alpha.grid(column = 3, row = 0, sticky = tk.NSEW)
        spin_nt_alpha.grid(column = 4, row = 0, sticky = tk.NSEW)
        
        button_tag_to_lora.grid(column = 0, row = 11, sticky = tk.NSEW)

        
        print(float(spin_learning_rate.get()), 50.0, float(spin_repeats.get()), float(spin_batch.get()), float(spin_epochs.get()))
        print(float(spin_learning_rate.get()) * 50.0 * float(spin_repeats.get()) * float(spin_batch.get()) * float(spin_epochs.get()))
    
    
    def copy_unet(self, ):
        spin_unet_rate.set(spin_learning_rate.get())
    def copy_enco(self, ):
        spin_enco_rate.set(spin_learning_rate.get())
    def auto_set_rate(self, ):
        step = float(text_num_img.get()) * float(spin_repeats.get()) * float(spin_batch.get()) * float(spin_epochs.get())
        spin_learning_rate.set("{:.6f}".format(float(spin_goal_rate.get()) / step))
    
    def ref_path(self, slct):  
        if slct == "model":
            fTyp = [("", "*")]
            url = r"E:\model"
            iDirPath = filedialog.askopenfilename(filetype = fTyp, initialdir = url)
            print(iDirPath)
            text_model_path.delete(0,tk.END)
            text_model_path.insert(0,iDirPath)
        if slct == "input":
            url = r"E:\trainingData\tageImage"
            iDirPath = filedialog.askdirectory(initialdir = url)
            text_input_path.delete(0,tk.END)
            text_input_path.insert(0,iDirPath)
            
            if text_input_path.get() != "":
                files_file = sorted(glob.glob(text_input_path.get()+'\\*.jpg'), key = os.path.getsize)
            
                for file in files_file:
                    tmp = file.split(".")
                    print(tmp)
                    im = Image.open(file)
                    im.save(tmp[0]+'.png')
                    os.remove(file)
                
                files_file = sorted(glob.glob(text_input_path.get()+'\\*.png'), key = os.path.getsize)
                text_num_img.delete(0, tk.END)
                text_num_img.insert(0, len(files_file))
        if slct == "output":
            url = r"E:\lora"
            iDirPath = filedialog.askdirectory(initialdir = url)
            text_output_path.delete(0,tk.END)
            text_output_path.insert(0,iDirPath)
                

# E:/diviewers/sd-scripts/venv/Scripts/python.exe train_network.py 
# --pretrained_model_name_or_path=E:/model/2Danime.safetensors --output_dir=E:/lora --output_name=sincos 
# --dataset_config=E:/trainingData/dataSetConfig.toml --train_batch_size=2 --max_train_epochs=4 --resolution=640,512 
# --optimizer_type=AdamW8bit --learning_rate=0.0001 --unet_lr=0.0001 --text_encoder_lr=0.0001 --network_dim=4 --network_alpha=1 
# --enable_bucket --bucket_no_upscale --lr_scheduler=cosine_with_restarts --lr_scheduler_num_cycles=4 --lr_warmup_steps=500 
# --keep_tokens=1 --shuffle_caption --caption_dropout_rate=0.05 --save_model_as=safetensors --clip_skip=2 --seed=42
# --color_aug --xformers --mixed_precision=fp16 --network_module=networks.lora --persistent_data_loader_workers
    def tag_to_lora(self,):
        data_lines = "[general]\n"
        data_lines = data_lines + "[[datasets]]\n"
        data_lines = data_lines + "[[datasets.subsets]]\n"
        data_lines = data_lines + "image_dir = \"" + text_input_path.get() + "\"\n"
        data_lines = data_lines + "caption_extension = \".txt\"\n"
        data_lines = data_lines + "num_repeats = " + spin_repeats.get()
        
        path = "E:/trainingData/dataSetConfig.toml"
        with open(path, mode="w", encoding="cp932") as f:
                f.write(data_lines)
        
        trigger = text_input_path.get()
        trigger = trigger.split("/")
        trigger = trigger[-1]        
        
        data_lines = []
        data_lines.append("train_network.py")
        data_lines.append("--pretrained_model_name_or_path=" + text_model_path.get())
        # data_lines.append("--train_data_dir=" + text_input_path.get()) 
        data_lines.append("--output_dir=" + text_output_path.get())
        data_lines.append("--output_name="  + trigger)
        data_lines.append("--dataset_config=E:/trainingData/dataSetConfig.toml")
        data_lines.append("--train_batch_size="  + spin_batch.get())# 特徴を反映しやすくする
        data_lines.append("--max_train_epochs="  + spin_epochs.get())
        data_lines.append("--resolution="  + spin_res_x.get() + "," + spin_res_y.get())
        data_lines.append("--optimizer_type=AdamW8bit")
        data_lines.append("--learning_rate="  + spin_learning_rate.get())
        data_lines.append("--unet_lr="  + spin_unet_rate.get())
        data_lines.append("--text_encoder_lr="  + spin_enco_rate.get())
        data_lines.append("--network_dim="  + spin_nt_dim.get())# 数が多いほど表現力は増す
        data_lines.append("--network_alpha="  + spin_nt_alpha.get())# アンダーフローを防ぎ安定して学習する
        data_lines.append("--enable_bucket")
        data_lines.append("--bucket_no_upscale")
        data_lines.append("--lr_scheduler=cosine_with_restarts")
        data_lines.append("--lr_scheduler_num_cycles=4")
        data_lines.append("--lr_warmup_steps=500")
        data_lines.append("--keep_tokens=1")
        data_lines.append("--shuffle_caption")
        data_lines.append("--caption_dropout_rate=0.05")
        data_lines.append("--save_model_as=safetensors")
        data_lines.append("--clip_skip=2")
        data_lines.append("--seed=42")    
        data_lines.append("--color_aug")
        data_lines.append("--xformers")
        data_lines.append("--mixed_precision=fp16")
        data_lines.append("--network_module=networks.lora")
        data_lines.append("--persistent_data_loader_workers")
              
        env = os.environ.copy()
        env["PYTHONPATH"]="E:/diviewers/sd-scripts/venv\Scripts/python"
        
        COMMAND =["start", "E:/diviewers/sd-scripts/venv/Scripts/python"]
        COMMAND = COMMAND + data_lines
        print(COMMAND)
        
        # COMMAND = ["start", r"E:\diviewers\sd-scripts\venv\Scripts\python", r"E:\diviewers\sd-scripts\convert_diffusers20_original_sd.py", r"E:\model\2D\defo\test3-50.ckpt", r".\test2.safetensors", r"--v1", r"--reference_model", r"runwayml/stable-diffusion-v1-5"]
        p = subprocess.Popen(COMMAND, env=env, cwd="E:/diviewers/sd-scripts", shell=True)
        

    def combo_prompt_name_set(self, _):
        combo_prompt_cunt.current(combo_prompt_name.current())
    def button_prompt_to_tag_one(self, ):
        global tag_list
        global cunt
        global name
        
        tag_list.append(combo_prompt_name.get())
        
        combo_tag["values"] = tag_list
        indx = name.index(combo_prompt_name.get())
        name.pop(indx)
        cunt.pop(indx)
        combo_prompt_name["values"] = name
        combo_prompt_cunt["values"] = cunt
        
        combo_prompt_name.set("")
        combo_prompt_cunt.set("")
    
    def button_prompt_to_tag_all(self, ):
        global tag_list
        global cunt
        global name
        
        for p in combo_prompt_name["values"]:
            tag_list.append(p)
        
        combo_tag["values"] = tag_list
        
        name = []
        cunt = []
        
        combo_prompt_name["values"] = name
        combo_prompt_cunt["values"] = cunt
        
        combo_prompt_name.set("")
        combo_prompt_cunt.set("")
        
    def button_tag_to_prompt(self, ):
        global tag_list
        global tag_cunt
        global cunt
        global name
        
        tag_list.remove(combo_tag.get())
        
        combo_tag["values"] = tag_list
        
        indx = -1
        for n,c in tag_cunt:
            print(n, c)
            if n == combo_tag.get():
                indx = c
                break
        
        if indx == -1:
            print("nothing")
            
        else:
            num = 0    
            for c in cunt:
                if c < indx:
                    break
                else:
                    num = num + 1
                    
            name.insert(num, combo_tag.get())
            cunt.insert(num, indx)
            combo_prompt_name["values"] = name
            combo_prompt_cunt["values"] = cunt
        
        combo_tag.set("")
        
    def button_add_prompt(self, ):
        global tag_list
        
        tag_list.append(combo_tag.get())
        
        combo_tag["values"] = tag_list
        

    def edit_tag(self, ):
        print("edit_tag")
        path = text_input_path.get()
        
        files_file = sorted(glob.glob(path+'\\*.txt'), key = os.path.getsize)
        
        print(combo_tag["values"])
        
        for file_name in files_file:
            with open(file_name, encoding="cp932") as f:
                data_lines = f.read()
            data_lines = data_lines.split(", ")
            
            for t in combo_tag["values"]:
                for i, d in enumerate(data_lines):
                    if t == d:
                        data_lines.pop(i)
            tmp = ""            
            for d in data_lines:
                tmp = tmp + d + ", "
            
            data_lines = tmp[:2]
            
            print(data_lines)
            with open(file_name, mode="w", encoding="cp932") as f:
                f.write(data_lines)
        

    def image_to_tag(self, ):

        path = text_input_path.get()
        
        
        files_file = sorted(glob.glob(path+'\\*.jpg'), key = os.path.getsize)
        
        for file in files_file:
            tmp = file.split(".")
            print(tmp)
            im = Image.open(file)
            im.save(tmp[0]+'.png')
            os.remove(file)
        
        files_file = sorted(glob.glob(path+'\\*.png'), key = os.path.getsize)
        
        
        COMMAND = "python tag_images_by_wd14_tagger.py  --batch_size 5 " + "\"" + path + "\""
        print(COMMAND)
        
        # p = subprocess.run(COMMAND, encoding='utf-8', stdin=sys.stdin, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        p = subprocess.run(COMMAND)
        
        files_file = sorted(glob.glob(path+'\\*.txt'), key = os.path.getsize)
        
        tag_all = ""
        
        trigger = path.split("/")
        
        for file_name in files_file:
            with open(file_name, encoding="cp932") as f:
                data_lines = f.read()
        
            data_lines = trigger[-1] + ", " + data_lines
            data_lines = data_lines.replace("\n", "")
            tag_all = tag_all + ", " + data_lines
        
            with open(file_name, mode="w", encoding="cp932") as f:
                f.write(data_lines)
        
        
        
        tag_all = tag_all.split(", ")
        tag_smpl = list(set(tag_all))
        tag_smpl.remove("")
        
        
        global tag_cunt
        global cunt
        global name
        tag_cunt = {}
        for t in tag_smpl:
            tag_cunt[t] = tag_all.count(t)
        
        # for n in range(len(tag_cunt)):
        #     print(n, tag_smpl[n], tag_cunt[n])
        
        tag_cunt = sorted(tag_cunt.items(), key=lambda x:x[1], reverse = True)
        # tag_cunt = dict((x, y) for x, y in tag_cunt)
        
        print(tag_cunt)
        # for t in tag_cunt.keys():
        #     print(t, tag_cunt[t])
        
        name = [x for x, _ in tag_cunt]
        cunt = [y for _, y in tag_cunt]
        combo_prompt_name["values"] = name
        combo_prompt_cunt["values"] = cunt
                            
        print("finish to tag")
                


    
class GNIMG():
    def __init__(self, root):
        global frame_menu
        global confData
        global cd
        global button_generate
        global combo_custom_set
        global combo_model
        global combo_scheduler
        global combo_vae
        global combo_positive_category
        global combo_negative_category
        global combo_lora
        global text_positive
        global spin_positive_insert
        global text_negative
        global spin_negative_insert
        global spin_lora
        global spin_lora_insert
        global spin_width
        global spin_height
        global spin_seed
        global spin_scale
        global spin_step
        global spin_times
        global label_time
        global text_log
        global combo_seed
        global combo_scale
        global combo_step
        global var_model_load
        global check_model_load
        global prgrs_var
        global prgrs_load
        
        global sub_view
        global sub_view_canvas
        global sub_lineup_img
        global sub_view_img
        global sub_view_id
        
        global sub_lineup
        global sub_lineup_canvas
        global sub_lineup_data
        global sub_lineup_stuck
        global sub_lineup_id
        global sub_lineup_id2
        
        global lineup_files
        
        global logline
        
        global clnt_set
        clnt_set = "test"
            
        confData = open("confData.json", "r")
        cd = json.load(confData)
        cd["list_custom_set"].sort()
        cd["list_lora"] = os.listdir("E:/lora/")
        cd["list_lora"].sort()
        # cd["list_lora"] = [f for f in cd["list_lora"] if f.find(".safetensors") != -1 or f.find(".pt") != -1]
        for n,f in enumerate(cd["list_lora"]):
            if f.find(".safetensors") != -1 or f.find(".pt") != -1:
                if len(cd["list_lora_extension"]) < n+1 :
                    cd["list_lora_extension"].append("")
                tmp = f.split(".")
                cd["list_lora"][n] = tmp[0]
                cd["list_lora_extension"][n] = "."+tmp[1]
                
        # cd["list_lora"] = [f.replace(".safetensors", "") for f in cd["list_lora"]]
        
        for lora in list(cd["list_lora_preset"].keys()):
            if lora not in cd["list_lora"]:
                cd["list_lora_preset"].pop(lora)
            
        width_column = [8, 34, 3, 2, 20, 70, 9]
        height_row = [5,8,10]
         
        root.title("diviewers")
        root.geometry("+0+0")
        root.bind("<FocusIn>", self.win_activate)
            
        frame_menu = tk.Frame(root)
        
        frame_custom_set = tk.Frame(frame_menu)
        button_custom_set = tk.Button(
            frame_custom_set,
            text = "Custom Set",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            )
        combo_custom_set = ttk.Combobox(
            frame_custom_set,
            values = cd["list_custom_set"],
            width = width_column[1],
            )
        combo_custom_set.bind("<<ComboboxSelected>>",self.set_clnt_set)  
        files = os.listdir("E:\\model")
        files_file = [f for f in files if f.find(".safetensors") != -1]
        files_file = [f.replace(".safetensors", "") for f in files_file]
        files_file.sort()
        
        button_custom_set_add = tk.Button(
            frame_custom_set,
            text = "＋",
            bg = "white",
            fg = "black",
            width = width_column[3],
            command = self.add_custom_set
            )
        button_custom_set_rmv = tk.Button(
            frame_custom_set,
            text = "ー",
            bg = "white",
            fg = "black",
            width = width_column[3],
            command = self.rmv_custom_set
            )
        
        frame_model = tk.Frame(frame_menu)
        button_model = tk.Button(
            frame_model,
            text = "Model",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            )
        combo_model = ttk.Combobox(
            frame_model,
            values = files_file,
            width = width_column[1],
            )
        var_model_load = tk.BooleanVar(frame_model)
        var_model_load.set(False)
        check_model_load = tk.Checkbutton(
            frame_model,
            text = "Load Model",
            variable = var_model_load,
            )
        
        frame_scheduler = tk.Frame(frame_menu)
        button_scheduler = tk.Button(
            frame_scheduler,
            text = "Scheduler",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            )
        combo_scheduler = ttk.Combobox(
            frame_scheduler,
            values = cd["list_scheduler"],
            width = width_column[1],
            )
        button_save_setting = tk.Button(
            frame_scheduler,
            text = "Save Set",
            bg = "white",
            fg = "black",
            width = width_column[0],
            command = self.save_setting,
            )
        
        frame_vae = tk.Frame(frame_menu)
        button_vae = tk.Button(
            frame_vae,
            text = "VAE",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            )
        combo_vae = ttk.Combobox(
            frame_vae,
            values = cd["list_vae"],
            width = width_column[1],
            )
        button_generate = tk.Button(
            frame_vae,
            text = "　▶︎",
            bg = "white",
            fg = "#009900",
            width = width_column[0],
            relief = tk.RAISED,
            # command = lambda: self.generate_switch("txt2img", ""),
            command = lambda args=("txt2img",""): self.generate_switch("event", args)
            )
        
        frame_positive = tk.Frame(frame_menu)
        button_positive = tk.Button(
            frame_positive,
            text = "Positive",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            )
        combo_positive_category = ttk.Combobox(
            frame_positive,
            width = width_column[4],
            )
        combo_positive_category.bind("<<ComboboxSelected>>",self.combo_positive_category_set)
        spin_positive_insert = ttk.Spinbox(
            frame_positive,
            width = width_column[2],
            from_ = 0,
            to = 9999,
            increment = 1,
            )
        button_positive_add = tk.Button(
            frame_positive,
            text = "＋",
            bg = "white",
            fg = "black",
            width = width_column[3],
            command = lambda: self.add_prompt(combo_positive_category.get(), "positive")
            )
        button_positive_rmv = tk.Button(
            frame_positive,
            text = "ー",
            bg = "white",
            fg = "black",
            width = width_column[3],
            command = lambda: self.rmv_prompt(combo_positive_category.get(), "positive")
            )
        button_positive_insert = tk.Button(
            frame_positive,
            text = "insert",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            command = lambda: self.positive_insert_prompt()
            )
        
        text_positive = scrolledtext.ScrolledText(
            frame_menu,
            width = width_column[5],
            height = height_row[0],
            undo=True,
            )
        
        frame_negative = tk.Frame(frame_menu)
        button_negative = tk.Button(
            frame_negative,
            text = "Negative",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            )
        combo_negative_category = ttk.Combobox(
            frame_negative,
            width = width_column[4],
            )
        combo_negative_category.bind(
            "<<ComboboxSelected>>",
            self.combo_negative_category_set
            )
        spin_negative_insert = ttk.Spinbox(
            frame_negative,
            width = width_column[2],
            from_ = 0,
            increment = 1,
            )
        button_negative_add = tk.Button(
            frame_negative,
            text = "＋",
            bg = "white",
            fg = "black",
            width = width_column[3],
            command = lambda: self.add_prompt(combo_negative_category.get(), "negative"),
            )
        button_negative_rmv = tk.Button(
            frame_negative,
            text = "ー",
            bg = "white",
            fg = "black",
            width = width_column[3],
            command = lambda: self.rmv_prompt(combo_negative_category.get(), "negative"),
            )
        button_negative_insert = tk.Button(
            frame_negative,
            text = "insert",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            command = lambda: self.negative_insert_prompt()
            )
        
        text_negative = scrolledtext.ScrolledText(
            frame_menu,
            width = width_column[5],
            height = height_row[0],
            undo = True,
            )
        
        frame_lora = tk.Frame(frame_menu)
        button_lora = tk.Button(
            frame_lora,
            text = "LoRA",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            command = lambda: self.regi_lora(),
            )
        combo_lora = ttk.Combobox(
            frame_lora,
            values = cd["list_lora"],
            width = width_column[1]-12,
            )
        combo_lora.bind(
            "<<ComboboxSelected>>",
            self.img_lora
            )
        spin_lora_insert = ttk.Spinbox(
            frame_lora,
            width = width_column[2],
            from_ = 0,
            to = 20,
            increment = 1,
            command = lambda: self.dsply_lora(), 
            )
        spin_lora = ttk.Spinbox(
            frame_lora,
            width = width_column[0]-2,
            from_ = -2,
            to = 2,
            increment = 0.05,
            )
        button_lora_add = tk.Button(
            frame_lora,
            text = "＋",
            bg = "white",
            fg = "black",
            width = width_column[3],
            command = lambda: self.add_lora(),
            )
        button_lora_rmv = tk.Button(
            frame_lora,
            text = "ー",
            bg = "white",
            fg = "black",
            width = width_column[3],
            command = lambda: self.rmv_lora(),
            )
        button_create_lora = tk.Button(
            frame_lora,
            text = "Create LoRA",
            bg = "white",
            fg = "black",
            width = width_column[0],
            command = lambda: self.switch_menu(),
            )
        button_lora_tag = tk.Button(
            frame_lora,
            text = "tagger",
            bg = "white",
            fg = "black",
            width = width_column[0],
            command = lambda: threading.Thread(target=self.image_to_tag).start()
            )
        button_lora_make = tk.Button(
            frame_lora,
            text = "make",
            bg = "white",
            fg = "black",
            width = width_column[0],
            command = lambda: threading.Thread(target=self.make_lora).start()
            )
        
        frame_others = tk.Frame(frame_menu)
        
        frame_size = tk.Frame(frame_others)
        button_size = tk.Button(
            frame_size,
            text = "Size",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            )
        label_width = tk.Label(
            frame_size,
            text = "W",
            width = width_column[2],        
            )
        label_height = tk.Label(
            frame_size,
            text = "H",
            width = width_column[2],        
            )
        spin_width = ttk.Spinbox(
            frame_size,
            from_ = 128,
            to = 2048,
            increment = 128,
            width = width_column[2]+1,
            )
        spin_height = ttk.Spinbox(
            frame_size,
            from_ = 128,
            to = 2048,
            increment = 128,
            width = width_column[2]+1,
            )
        
        text_log = scrolledtext.ScrolledText(
            frame_others,
            width = 31,
            height = 4,
            )
        
        frame_seed = tk.Frame(frame_others)
        button_seed = tk.Button(
            frame_seed,
            text = "Seed",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            )
        spin_seed = ttk.Spinbox(
            frame_seed,
            from_ = -1,
            to = 2**31,
            increment = 10,
            width = width_column[6],
            )
        combo_seed = ttk.Combobox(
            frame_seed,
            values = cd["list_seed"],
            width = width_column[0],
            )
        
        frame_scale = tk.Frame(frame_others)
        button_scale = tk.Button(
            frame_scale,
            text = "Scale",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            )
        spin_scale = ttk.Spinbox(
            frame_scale,
            from_ = 0,
            to = 32,
            increment = 1,
            width = width_column[6],
            )
        combo_scale = ttk.Combobox(
            frame_scale,
            values = cd["list_scale"],
            width = width_column[0],
            )
        
        frame_step = tk.Frame(frame_others)
        button_step = tk.Button(
            frame_step,
            text = "Step",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            )
        spin_step = ttk.Spinbox(
            frame_step,
            from_ = 0,
            to = 100,
            increment = 10,
            width = width_column[6],
            )
        combo_step = ttk.Combobox(
            frame_step,
            values = cd["list_step"],
            width = width_column[0],
            )
        
        frame_times = tk.Frame(frame_others)
        button_times = tk.Button(
            frame_times,
            text = "Times",
            bg = "white",
            fg = "black",
            width = width_column[0],
            relief = tk.RAISED,
            )
        spin_times = ttk.Spinbox(
            frame_times,
            from_ = 1,
            to = 99999,
            increment = 1,
            width = width_column[6],
            )
        label_time = tk.Label(
            frame_times,
            text = "00.00",
            width = width_column[0]+1,        
            )
        prgrs_var = tk.IntVar()        
        prgrs_load = ttk.Progressbar(
            frame_times,
            maximum=100,
            length=258,
            mode="determinate",
            variable=prgrs_var,
            )
        
        
        button_test = tk.Button(
            frame_lora,
            text = "test",
            bg = "white",
            fg = "black",
            relief = tk.RAISED,
            command = lambda: self.test(),
            )


        frame_menu.grid(column = 0, row = 0)
        
        frame_custom_set.grid(column = 0, row = 0, sticky = tk.NSEW)
        button_custom_set.grid(column = 0, row = 0, sticky = tk.NSEW)
        combo_custom_set.grid(column = 1, row = 0, sticky = tk.NSEW)
        button_custom_set_add.grid(column = 2, row = 0, sticky = tk.NSEW)
        button_custom_set_rmv.grid(column = 3, row = 0, sticky = tk.NSEW)
        
        frame_model.grid(column = 0, row = 1, sticky = tk.NSEW)
        button_model.grid(column = 0, row = 0, sticky = tk.NSEW)
        combo_model.grid(column = 1, row = 0, sticky = tk.NSEW)
        check_model_load.grid(column = 2, row = 0, sticky = tk.NSEW)
        
        frame_scheduler.grid(column = 0, row = 2, sticky = tk.NSEW)
        button_scheduler.grid(column = 0, row = 0, sticky = tk.NSEW)
        combo_scheduler.grid(column = 1, row = 0, sticky = tk.NSEW)
        button_save_setting.grid(column = 2, row = 0, sticky = tk.NSEW)
        
        frame_vae.grid(column = 0, row = 3, sticky = tk.NSEW)
        button_vae.grid(column = 0, row = 0, sticky = tk.NSEW)
        combo_vae.grid(column = 1, row = 0, sticky = tk.NSEW)
        button_generate.grid(column = 2, row = 0, sticky = tk.NSEW)
        
        frame_positive.grid(column = 0, row = 4, sticky = tk.NSEW)
        button_positive.grid(column = 0, row = 0, sticky = tk.NSEW)
        combo_positive_category.grid(column = 1, row = 0, sticky = tk.NSEW)
        spin_positive_insert.grid(column = 2, row = 0, sticky = tk.NSEW)
        button_positive_add.grid(column = 3, row = 0, sticky = tk.NSEW)
        button_positive_rmv.grid(column = 4, row = 0, sticky = tk.NSEW)
        button_positive_insert.grid(column = 5, row = 0, sticky = tk.NSEW)
        
        text_positive.grid(column = 0, row = 5, sticky = tk.NSEW)
        
        frame_negative.grid(column = 0, row = 6, sticky = tk.NSEW)
        button_negative.grid(column = 0, row = 0, sticky = tk.NSEW)
        combo_negative_category.grid(column = 1, row = 0, sticky = tk.NSEW)
        spin_negative_insert.grid(column = 2, row = 0, sticky = tk.NSEW)
        button_negative_add.grid(column = 3, row = 0, sticky = tk.NSEW)
        button_negative_rmv.grid(column = 4, row = 0, sticky = tk.NSEW)
        button_negative_insert.grid(column = 5, row = 0, sticky = tk.NSEW)
        
        text_negative.grid(column = 0, row = 7, sticky = tk.NSEW)
        
        frame_lora.grid(column = 0, row = 8, sticky = tk.NSEW)
        button_lora.grid(column = 0, row = 0, sticky = tk.NSEW)
        combo_lora.grid(column = 1, row = 0, sticky = tk.NSEW)
        spin_lora_insert.grid(column = 2, row = 0, sticky = tk.NSEW)
        spin_lora.grid(column = 3, row = 0, sticky = tk.NSEW)
        button_lora_add.grid(column = 4, row = 0, sticky = tk.NSEW)
        button_lora_rmv.grid(column = 5, row = 0, sticky = tk.NSEW)
        button_create_lora.grid(column = 6, row = 0, sticky = tk.NSEW)
        
        
                
        frame_others.grid(column = 0, row = 9, sticky = tk.NSEW)
        
        frame_size.grid(column = 0, row = 0, sticky = tk.NSEW)
        button_size.grid(column = 0, row = 0, sticky = tk.NSEW)
        label_width.grid(column = 1, row = 0, sticky = tk.NSEW)
        spin_width.grid(column = 2, row = 0, sticky = tk.NSEW)
        label_height.grid(column = 3, row = 0, sticky = tk.NSEW)
        spin_height.grid(column = 4, row = 0, sticky = tk.NSEW)
        
        text_log.grid(column = 1, row = 0, sticky = tk.NSEW, rowspan = 4)
        
        frame_seed.grid(column = 0, row = 1, sticky = tk.NSEW)
        button_seed.grid(column = 0, row = 0, sticky = tk.NSEW)
        spin_seed.grid(column = 1, row = 0, sticky = tk.NSEW)
        combo_seed.grid(column = 2, row = 0, sticky = tk.NSEW)
        
        
        frame_scale.grid(column = 0, row = 2, sticky = tk.NSEW)
        button_scale.grid(column = 0, row = 0, sticky = tk.NSEW)
        spin_scale.grid(column = 1, row = 0, sticky = tk.NSEW)
        combo_scale.grid(column = 2, row = 0, sticky = tk.NSEW)
        
        frame_step.grid(column = 0, row = 3, sticky = tk.NSEW)
        button_step.grid(column = 0, row = 0, sticky = tk.NSEW)
        spin_step.grid(column = 1, row = 0, sticky = tk.NSEW)
        combo_step.grid(column = 2, row = 0, sticky = tk.NSEW)
        
        frame_times.grid(column = 0, row = 4, sticky = tk.NSEW, columnspan = 2)
        button_times.grid(column = 0, row = 0, sticky = tk.NSEW)
        spin_times.grid(column = 1, row = 0, sticky = tk.NSEW)
        label_time.grid(column = 2, row = 0, sticky = tk.NSEW)
        prgrs_load.grid(column = 3, row = 0, sticky = tk.NSEW)
        
        
        # button_test.grid(column = 4, row = 0, sticky = tk.NSEW)
        
        
        sub_view = tk.Toplevel()
        sub_lineup = tk.Toplevel()
        
        sub_view.geometry("1024x1024+870+30")
        
        sub_view_canvas = tk.Canvas(sub_view, highlightthickness=0, relief="ridge", bg = "black", height = 1024, width = 1024)
        sub_view_id = sub_view_canvas.create_image(0,0, anchor = tk.NW)
        sub_view_canvas.grid(column = 0, row = 0, sticky = tk.NSEW)
        
        sub_view.bind("<Button-1>", self.lift_window)
        sub_view.bind("<ButtonRelease-2>", lambda event, arg=0: self.change_sub_view_canvas(event, arg))
        sub_view.bind("<KeyPress>", self.select_image)
        
        

        sub_lineup["height"] = 1000
        sub_lineup["width"] = 1200
        sub_lineup.geometry("+534+30")
        sub_lineup.update()
 
        sub_lineup_canvas = []
        sub_lineup_id = []
        
        n = 0
        for x in range(4):
            for y in range(3):
                if n % 2 == 0:
                    c = "white"
                else:
                    c = "black"
        
                sub_lineup_canvas.append(tk.Canvas(sub_lineup, highlightthickness=0, relief="ridge"))
                sub_lineup_canvas[n]["bg"] = c
                sub_lineup_id.append(sub_lineup_canvas[n].create_image(0, 0, anchor = tk.NW))
                sub_lineup_canvas[n].grid(column = x, row = y,sticky = tk.NSEW)
                sub_lineup_canvas[n].bind("<Button-1>", lambda event, arg=n: self.change_sub_view_canvas(event, arg))
                sub_lineup_canvas[n].bind("<Button-2>", lambda event, arg=("img2img", n): self.generate_switch(event, arg))
                sub_lineup_canvas[n].bind("<Button-3>", lambda event, arg=n: self.delete_sub_lineup_canvas(event, arg))
        
                n += 1
        
        self.change_size(0)
        sub_lineup.bind("<ButtonRelease-2>", self.change_size)
        
        sub_view_img = sub_lineup_data[0]
        
        root.protocol("WM_DELETE_WINDOW", gngui.quit)
    
    def click_close(self, ):
        global gngui
        
        # sub_view.destroy()
        # sub_lineup.destroy()
        gngui.quit()
        
    
    
    def lift_window(self, _):
        sub_lineup.lift()
    
    def select_image(self, event):
        global crnt_index
        
        key_name = event.keysym
        
        if key_name == "Left":
            print("left")
        elif key_name == "Right":
            print("right")
        
    
    def img_lora(self, _):
        global sub_view_id
        global sub_view_img
        
        win_height = sub_view.winfo_height()
        win_width = sub_view.winfo_width()

        path = "E:/lora_img/"
        
        if os.path.isfile(path + combo_lora.get() + ".jpeg"):
            file = path + combo_lora.get() + ".jpeg"
        elif os.path.isfile(path + combo_lora.get() + ".png"):
            file = path + combo_lora.get() + ".png"
        else:
            file = path + "null_img.png"
        
        file = Image.open(file)
       
        if file.height < file.width:
            rasio = file.width / win_width
        else:
            rasio = file.height / win_height
        
        cnvs_height = int(file.height / rasio)
        cnvs_width = int(file.width / rasio)
        
        sub_view_img = file.resize((cnvs_width, cnvs_height))
        sub_view_img = ImageTk.PhotoImage(sub_view_img)
        
        # sub_view_img["width"] = cnvs_width
        # sub_view_img["height"] = cnvs_height
        
        sub_view_canvas["width"] = cnvs_width
        sub_view_canvas["height"] = cnvs_height
        
        sub_view_canvas.itemconfig(sub_view_id, image = sub_view_img)
        sub_view.update()
        
        sub_view.lift()
        
    def change_sub_view_canvas(self, _, num):
        global sub_lineup
        global sub_lineup_data
        global sub_view_img
        global sub_view_canvas
        global sub_view_id
        
        win_height = sub_view.winfo_height()
        win_width = sub_view.winfo_width()
        
        sub_view_img = sub_lineup_data[num]
        
        if win_height < win_width:
            rasio = win_height / sub_view_img.height
        else:
            rasio = win_width / sub_view_img.width
            
        win_height = int(sub_view_img.height * rasio)
        win_width = int(sub_view_img.width * rasio)
        
        sub_view_img = sub_view_img.resize((win_width, win_height))
        sub_view_img = ImageTk.PhotoImage(sub_view_img)
        
        sub_view_canvas["height"] = win_height
        sub_view_canvas["width"] = win_width
        
        sub_view["height"] = win_height
        sub_view["width"] = win_width
        
        sub_view_canvas.itemconfig(sub_view_id, image = sub_view_img)
        
        sub_view.update()
        
        sub_view.lift()
        
    def delete_sub_lineup_canvas(self, _, num):
        global lineup_files
        
        os.remove(lineup_files[num])
        
        self.change_size(0)
     
    def change_size(self, _):
        global sub_lineup
        global sub_lineup_data
        global sub_lineup_img
        global lineup_files
        
        sep_x = 4
        sep_y = 3

        win_height = sub_lineup.winfo_height()
        win_width = sub_lineup.winfo_width()
        max_width = 0
        max_height = 0
       
        lineup_files = glob.glob("E:/output/*.png")
        lineup_files.sort(key = os.path.getmtime, reverse = True)
        sub_lineup_data = [Image.open(lineup_files[f]) for f in range(4*3)]
       
        for n in range(sep_x*sep_y-1):
            if max_width < sub_lineup_data[n].width:
                max_width = sub_lineup_data[n].width
            if max_height < sub_lineup_data[n].height:
                max_height = sub_lineup_data[n].height
                        
        if win_height < win_width:
            rasio = max_height * sep_y / win_height
        else:
            rasio = max_width * sep_x / win_width
        
        cnvs_height = int(max_height / rasio)
        cnvs_width = int(max_width / rasio)
        
        sub_lineup_img = [sub_lineup_data[n].resize((cnvs_width, cnvs_height)) for n in range(4*3)]
        sub_lineup_img = [ImageTk.PhotoImage(f) for f in sub_lineup_img]
        
        n = 0
        for _ in range(4):
            for _ in range(3):                   
                if n < len(sub_lineup_img):
                    sub_lineup_canvas[n]["width"] = cnvs_width
                    sub_lineup_canvas[n]["height"] = cnvs_height

                    sub_lineup_canvas[n].itemconfig(sub_lineup_id[n], image = sub_lineup_img[n])
                
                n += 1
        sub_lineup.update()        
    
    def log_code(self):
        global ferr
        
        while True:
            try:
                ferr_in = ferr.getvalue()
                output_sp = ferr_in.split("<")
                output_sp = output_sp[-1].split(",")
                
                label_time["text"] = output_sp[0]
                
                ferr_in = ferr.getvalue()
                output_sp = ferr_in.split("\r")
                output_sp = output_sp[-1].split("%")
                output_sp[0] = output_sp[0].replace(" ", "")
                
                prgrs_var.set(int(output_sp[0]))
                
                if output_sp[0] == "100":
                    break
                    
            except:
                _ = 0
            
            sleep(0.1)
        
        
    def log_prcs(self, in_file, out_file):
        global ferr
        global ferr_pre
        global ferr_gap
        global fout
        global fout_pre
        global fout_gap
        
        global log_line
        global prg_generate
        global cp_generate
        global prgrs_var
        global prgrs_load
        global label_time
        
        log_line = []
        prg_generate = -1
        cp_generate = 0
        
        ferr_pre = ""
        fout_pre = ""
        
        while True:
            try:
                for line in in_file:
                    print(line.strip(), flush=True)
                
                    if "%|" in line:                        
                        output_sp = line.strip()
                        output_sp = output_sp.split("<")
                        output_sp = output_sp[-1].split(",")
                        
                        label_time["text"] = output_sp[0]       
                        
                        output_sp = line.strip()
                        output_sp = output_sp.split("%|")
                        # output_sp[0] = output_sp[0].replace(" ", "")
                        print(output_sp[0], flush=True)
                        prgrs_var.set(int(output_sp[0]))
                        
                if output_sp[0] == "100":
                    break
        
                
            except:
                _ = 0
            
            sleep(0.1)

        
    def set_clnt_set(self, e):
        global clnt_set
        clnt_set = combo_custom_set.get()
        self.dsply_custom(combo_custom_set.get())

    def dsply_custom(self, e):        
        print("dsply_custom :", combo_custom_set.get())
        self.dsply_log("END", "display custom setting:"+combo_custom_set.get())
        cs = cd["list_detail"][combo_custom_set.get()]
    
        combo_model.set(cs["model"])
        combo_scheduler.set(cs["scheduler"])
        combo_vae.set(cs["vae"])
        
        combo_positive_category["values"] = cs["prompt"]["positive"]["category"]
        combo_negative_category["values"] = cs["prompt"]["negative"]["category"]
        
        spin_width.set(cs["size"]["width"])
        spin_height.set(cs["size"]["height"])
        
        spin_seed.set(cs["seed"]["number"])
        combo_seed.set(cs["seed"]["shift"])
        spin_times.set(cs["seed"]["times"])
        
        spin_scale.set(cs["scale"]["number"])
        combo_scale.set(cs["scale"]["shift"])
        
        spin_step.set(cs["step"]["number"])
        combo_step.set(cs["step"]["shift"])
        
    def dsply_clear(self):
        combo_model.set("")
        combo_scheduler.set("")
        combo_vae.set("")
        
        combo_positive_category.set("")
        text_positive.delete("1.0", tk.END)
        combo_negative_category.set("")
        text_negative.delete("1.0", tk.END)
        
        combo_lora.set("")
        spin_lora.set("")
        
        spin_width.set("")
        spin_height.set("")
        spin_seed.set("")
        spin_scale.set("")
        spin_step.set("")
        combo_seed.set("")
        combo_scale.set("")
        combo_step.set("")
        
            
            
    def combo_positive_category_set(self, e): 
        cs = cd["list_detail"][combo_custom_set.get()]["prompt"]["positive"]
        factor = combo_positive_category.get()
        if factor in cs["category"]:
            index = cs["category"].index(factor)
            text_positive.delete("1.0", tk.END)
            text_positive.insert("1.0", cs["spell"][index])
            spin_positive_insert.set(index)
    
    def combo_negative_category_set(self, e): 
        cs = cd["list_detail"][combo_custom_set.get()]["prompt"]["negative"]
        factor = combo_negative_category.get()
        if factor in cs["category"]:
            index = cs["category"].index(factor)
            text_negative.delete("1.0", tk.END)
            text_negative.insert("1.0", cs["spell"][index])
            spin_negative_insert.set(index)
        
    def positive_insert_prompt(self):
        cs = cd["list_detail"][combo_custom_set.get()]["prompt"]["positive"]
        
        factor = combo_positive_category.get()
        spell = text_positive.get("1.0", tk.END)
        
        if factor in cs["category"]:
            index = cs["category"].index(factor)
            cs["category"].pop(index)
            cs["spell"].pop(index)
            
            index = spin_positive_insert.get()
            if index == "":
                cs["category"].append(factor)
                cs["spell"].append(spell)
            else:
                index = int(index)
                cs["category"].insert(index, factor)
                cs["spell"].insert(index, spell)
            
            combo_positive_category["values"] = cs["category"]
        
    def negative_insert_prompt(self):    
        cs = cd["list_detail"][combo_custom_set.get()]["prompt"]["negative"]
        
        factor = combo_negative_category.get()
        spell = text_negative.get("1.0", tk.END)
        
        if factor in cs["category"]:
            index = cs["category"].index(factor)
            cs["category"].pop(index)
            cs["spell"].pop(index)
            
            index = spin_negative_insert.get()
            if index == "":
                cs["category"].append(factor)
                cs["spell"].append(spell)
            else:
                index = int(index)
                cs["category"].insert(index, factor)
                cs["spell"].insert(index, spell)
            
            combo_negative_category["values"] = cs["category"]
    
    def add_custom_set(self):
        global clnt_set
        if combo_custom_set.get() not in cd["list_custom_set"]:
            cd["list_custom_set"].append(combo_custom_set.get())
            cd["list_detail"][combo_custom_set.get()] = {"","","","","","","",""}
            if clnt_set in cd["list_custom_set"]: 
                cd["list_detail"][combo_custom_set.get()] = copy.deepcopy(cd["list_detail"][clnt_set])
                clnt_set = combo_custom_set.get()
                combo_custom_set.set(clnt_set)
            else:
                cd["list_detail"][combo_custom_set.get()] = copy.deepcopy(cd["list_detail"]["2D_template"])
            
            self.dsply_clear()
            self.dsply_custom(clnt_set)
            
    def rmv_custom_set(self):
        if combo_custom_set.get() in cd["list_custom_set"]:
            del cd["list_detail"][combo_custom_set.get()]
            cd["list_custom_set"].remove(combo_custom_set.get())
            combo_custom_set["value"] = cd["list_custom_set"]
            combo_custom_set.set("")

            self.dsply_clear()
        
    def add_prompt(self, factor, tive):
        print("add prompt" , factor, ":", combo_custom_set.get())
        self.dsply_log("END", "add prompt:"+factor)
        global cd
        cs =  cd["list_detail"][combo_custom_set.get()]["prompt"]
        if tive == "positive":
            spell = text_positive.get("1.0", tk.END)
            index = spin_positive_insert.get()
            spin_positive_insert.set("")
        else:
            spell = text_negative.get("1.0", tk.END)
            index = spin_negative_insert.get()
            spin_negative_insert.set("")
        
        if factor in cs[tive]["category"]:
            index = cs[tive]["category"].index(factor)
            cs[tive]["spell"][index] = spell
            if cs[tive]["spell"][index][-1] == "\n":
                cs[tive]["spell"][index] = cs[tive]["spell"][index][0:-1]
        else:
            if index == "":
                cs[tive]["category"].append(factor)
                cs[tive]["spell"].append(spell)
            else:
                index = int(index)
                cs[tive]["category"].insert(index, factor)
                cs[tive]["spell"].insert(index, spell)
                       
        if tive == "positive":
            combo_positive_category["values"] = cs["positive"]["category"]
            combo_positive_category.set("")
            text_positive.delete("1.0", tk.END)
        else:
            combo_negative_category["values"] = cs["negative"]["category"]
            combo_negative_category.set("")
            text_negative.delete("1.0", tk.END)
            
        # pprint.pprint(cd, sort_dicts=False)
                        
    def rmv_prompt(self, factor, tive):
        cs =  cd["list_detail"][combo_custom_set.get()]["prompt"]
        
        if factor in cs[tive]["category"]:
            index = cs[tive]["category"].index(factor)
            cs[tive]["category"].pop(index)
            cs[tive]["spell"].pop(index)
        
        if tive == "positive":
            combo_positive_category["values"] = cs["positive"]["category"]
            combo_positive_category.set("")
            text_positive.delete("1.0", tk.END)
        else:
            combo_negative_category["valuse"] = cs["negative"]["category"]
            combo_negative_category.set("")
            text_negative.delete("1.0", tk.END)
    
    def add_lora(self):
        cs = cd["list_detail"][combo_custom_set.get()]
        
        factor = combo_lora.get()
        if spin_lora.get() == "":
            alpha = "0.5"
            spin_lora.set("0.5")
        else:
            alpha = spin_lora.get()
        
        index = int(spin_lora_insert.get())
        
        if "textInv" in factor:
            print("add text inversion :", factor[8:])
            self.dsply_log("END", "add text inversion:"+factor[8:])
        else:
            print("add lora :", factor)
            self.dsply_log("END", "add lora:"+factor)
        
        lora_length = len(cs["lora"]["name"])
        if lora_length <= index:
            for i in range(index-lora_length+1):
                cs["lora"]["name"].append("")
                cs["lora"]["alpha"].append("")
                
        cs["lora"]["name"][index] = factor
        cs["lora"]["alpha"][index] = alpha
        
        lora_length = len(cs["lora"]["name"])
        for i in range(lora_length):
            j = lora_length - i - 1

            if cs["lora"]["name"][j] == "":
                cs["lora"]["name"].pop(j)
                cs["lora"]["alpha"].pop(j)
        
        if "textInv" in factor:
            combo_negative_category.set(cs["prompt"]["negative"]["category"][0])
            spin_negative_insert.set(0)
            text_negative.delete(1.0, tk.END)
            text_negative.insert(tk.END, cs["prompt"]["negative"]["spell"][0]+factor[8:]+", ")
        
        else:   
            combo_positive_category.set(factor)
            spin_positive_insert.set(0)
            
            if factor not in cd["list_lora_preset"]:
                cd["list_lora_preset"][factor] = []
                
            text_positive.delete(1.0, tk.END)
            text_positive.insert(tk.END, cd["list_lora_preset"][factor])
            
        
        self.dsply_lora()
        
    def regi_lora(self):
        cd["list_lora_preset"][combo_lora.get()] = text_positive.get(1.0, tk.END)
        self.save_setting()
                    
    def rmv_lora(self):
        cs = cd["list_detail"][combo_custom_set.get()]["lora"]
        
        index = int(spin_lora_insert.get())
        
        lora_length = len(cs["name"])
        if lora_length >= index:
            cs["name"][index] = ""
            cs["alpha"][index] = ""
        
        lora_length = len(cs["name"])
        for i in range(lora_length):
            j = lora_length - i - 1
            if cs["name"][j] == "":
                cs["name"].pop(j)
                cs["alpha"].pop(j)
        
        self.dsply_lora()
        
    def dsply_lora(self):
        cs = cd["list_detail"][combo_custom_set.get()]["lora"]
        index = int(spin_lora_insert.get())

        if len(cs["name"]) > index:
            combo_lora.set(cs["name"][index])
            spin_lora.set(cs["alpha"][index])
        else:
            combo_lora.set("")
            spin_lora.set("")
        
    def save_setting(self):
        print("Save setting :", combo_custom_set.get())
        self.dsply_log("END", "save setting:"+combo_custom_set.get())
        global cd
        if combo_custom_set.get() != "":
            cs = cd["list_detail"][combo_custom_set.get()]
            
            cs.update([("model", combo_model.get()), ("scheduler", combo_scheduler.get()), ("vae", combo_vae.get())])
            cs["size"].update([("width", spin_width.get()),("height", spin_height.get())])
            cs["seed"].update([("number", spin_seed.get()),("shift", combo_seed.get()),("times", spin_times.get())])
            cs["scale"].update([("number", spin_scale.get()),("shift", combo_scale.get())])
            cs["step"].update([("number", spin_step.get()),("shift", combo_step.get())])
        
        tmp = cd
        tmpData = open("confData.json", "w") 
        json.dump(tmp, tmpData, indent = 4)
        # print(json.dumps(tmp, indent=4))
    
    def generate_switch(self, _, args):
        print("generate_switch:", args)
        global cd
        global stdout_log
        global threading_generate
        if button_generate["text"] == "▶︎":
            f = False
            try:
                if threading_generate.is_alive():
                    print("threading_generate is alive")
                    self.dsply_log("END", "threading_generate is still alive.")
                else:
                    f = True
            except:
                print("threading_generate is not exist")
                f = True
                
            if f:
                button_generate["text"] = "◼︎︎"
                button_generate["fg"] = "#AA0000"
                threading_generate = threading.Thread(target=self.generate, args=(args[0], args[1]))
                
                threading_generate.start()
        else:
            button_generate["text"] = "▶︎"
            button_generate["fg"] = "#00AA00"

       
    def generate(self, tsk, img):
        global log_line
        
        log_line = []
        
        print("Start generating")
        self.dsply_log("END", "start generating:"+combo_custom_set.get())
        global sub_lineup_canvas
        global OUTPUT_PATH
        global SEED
        global MODEL_NAME
        global pipe
        global now
        
        CHECKPOINT_PATH = "E:\\model\\"
        TMP_MODEL_PATH = 
        LORA_PATH = "E:\\lora\\"
        ALPHA = "NA"
        OUTPUT_PATH = "E:\\output\\"
        DEVICE = "cuda"
        
        dt_now = datetime.datetime.now()
        now = dt_now.strftime("%H") + "：" + dt_now.strftime("%M") + "：" + dt_now.strftime("%S")
        
        SCHEDULERS_LIST = {
            'DDIM' : DDIMScheduler,
            'DDPM' : DDPMScheduler,
            'DPMSolverMultistep' : DPMSolverMultistepScheduler,
            'DPMSolverSinglestep' : DPMSolverSinglestepScheduler,
            'EulerAncestralDiscrete' : EulerAncestralDiscreteScheduler,
            'EulerDiscrete' : EulerDiscreteScheduler,
            'HeunDiscrete' : HeunDiscreteScheduler,
            'KDPM2AncestralDiscrete' : KDPM2AncestralDiscreteScheduler,
            'KDPM2Discrete' : KDPM2DiscreteScheduler,
            "LMSDiscrete" : LMSDiscreteScheduler,
            "PNDM" : PNDMScheduler,
        }


                
        # screen_pos = pyautogui.size()
        # center_x = screen_pos.width / 2
        # center_y = screen_pos.height / 2
        # pyautogui.moveTo(center_x,center_y)
        # x,y = pyautogui.position()
        
        seed_span = {"":0, "+ n":1, "+ n*10":10, "+ n*100":100}
        scale_span = {"":0, "+ n":1, "+ n*2":2}
        step_span = {"":0, "+ n*10":10}
        
        global STEPS
        global tp
        global sub_lineup_data
        global lineup_files
        global sub_lineup_id
    
        
        cs = cd["list_detail"][combo_custom_set.get()]
        
        if cs["seed"]["number"] == "-1":
            SEED = random.randrange(999999)
        else:
            SEED = int(cs["seed"]["number"]) 
        
        if tsk == "img2img":
            TIMES = 3
        else:
            TIMES = int(cs["seed"]["times"])
            
        for i in range(TIMES):
            if button_generate["text"] == "▶︎":
                break
            
            cs = cd["list_detail"][combo_custom_set.get()]
            MODEL_NAME = combo_model.get()
            
            SCHEDULER_NAME = cs["scheduler"]
            
            VAE_NAME = cs["vae"]
                    
            CHECKPOINT_ID = CHECKPOINT_PATH + MODEL_NAME
            TMP_MODEL_ID = TMP_MODEL_PATH + MODEL_NAME
               
            print("Check if there is a checkpoint")
            self.dsply_log("END", "check if there is a checkpoint.")
            
            if combo_model.get() not in cd["list_model_mtime"]:
                
                cd["list_model_mtime"][combo_model.get()] = "0"
            
            if  cd["list_model_mtime"][combo_model.get()] != os.path.getmtime(CHECKPOINT_ID+".safetensors") or var_model_load.get():
                
                pipe = StableDiffusionPipeline.from_ckpt(CHECKPOINT_ID+".safetensors", torch_dtype=torch.float16)
                pipe.save_pretrained(TMP_MODEL_ID, safe_serialization=True)
                cd["list_model_mtime"][combo_model.get()] = os.path.getmtime(CHECKPOINT_ID+".safetensors")
                self.save_setting()

            
            VAE = AutoencoderKL.from_pretrained(
                VAE_NAME,
                torch_dtype=torch.float16
                ).to(DEVICE)
                
            print("Model Path :", TMP_MODEL_ID)
            self.dsply_log("END", "set to pipe:"+MODEL_NAME)
            self.dsply_log("END", "set to pipe:"+VAE_NAME)
            
            if tsk == "txt2img":
                pipe = StableDiffusionPipeline.from_pretrained(
                    TMP_MODEL_ID, 
                    custom_pipeline="lpw_stable_diffusion",
                    vae=VAE,
                    safety_checker = None, 
                    # revision = "fp16",
                    torch_dtype=torch.float16,
                    ).to(DEVICE)
                    
            elif tsk == "img2img":        
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    TMP_MODEL_ID,
                    custom_pipeline="lpw_stable_diffusion",
                    vae=VAE,
                    safety_checker = None, 
                    # revision = "fp16",
                    torch_dtype=torch.float16
                    ).to(DEVICE)

                              
            for index in range(len(cs["lora"]["name"])):
                LORA_NAME = cs["lora"]["name"][index]
                
                LORA_ID = LORA_PATH + LORA_NAME + cd["list_lora_extension"][cd["list_lora"].index(LORA_NAME)]
                if "textInv" in LORA_NAME:
                    print("text inversion NAME", index, ":", LORA_NAME[8:])

                    self.dsply_log("END", "textInv:"+LORA_NAME[8:])
                    pipe.load_textual_inversion(
                        LORA_ID, 
                        token=LORA_NAME[8:]
                    )
                    
                else:
                    ALPHA = float(cs["lora"]["alpha"][index])
                    print("LoRA NAME", index, ":", LORA_NAME,ALPHA)
                    self.dsply_log("END", "LoRA:"+LORA_NAME+" "+str(ALPHA))
                    pipe = load_safetensors_lora(
                        pipe, 
                        LORA_ID,
                        float(ALPHA)
                    ).to(DEVICE)
            
            pipe.scheduler = SCHEDULERS_LIST[SCHEDULER_NAME].from_config(pipe.scheduler.config)
            
            
            pipe.vae.enable_tiling()
            
            
            POGPROMPT = ""
            tmp = ""
            for factor in cs["prompt"]["positive"]["spell"]:
                factor += "\n"
                while len(factor) > 1:
                    if factor[0] != "#":
                        POGPROMPT += factor[0:factor.find("\n")]
                    tmp = factor[factor.find("\n")+1:]
                    factor = tmp
                
                POGPROMPT = POGPROMPT.replace("\n", "").replace("  ", " ").replace(", ", ",").replace(" ","_")
            
            NEGPROMPT = ""
            tmp = ""
            for factor in cs["prompt"]["negative"]["spell"]:
                factor += "\n"
                while len(factor) > 1:
                    if factor[0] != "#":
                        NEGPROMPT += factor[0:factor.find("\n")]
                    tmp = factor[factor.find("\n")+1:]
                    factor = tmp
                
                NEGPROMPT = NEGPROMPT.replace("\n","").replace("  ", " ").replace(", ", ",").replace(" ","_")
            
            print("Generate No.",i+1)
            self.dsply_log("END", "generate:"+str(i+1)+"/"+str(cs["seed"]["times"]))
                       
            SCALE = int(cs["scale"]["number"]) + i * scale_span[cs["scale"]["shift"]]
            STEPS = int(cs["step"]["number"]) + i * step_span[cs["step"]["shift"]]
            WIDTH_SIZE = int(cs["size"]["width"])
            HEIGHT_SIZE = int(cs["size"]["height"])
            seed = torch.Generator(device=DEVICE).manual_seed(SEED)
            
            tp = STEPS / 3
            
            pipe.to(DEVICE)
            
            pipe.enable_attention_slicing()
            pipe.enable_xformers_memory_efficient_attention()
            torch.torch.backends.cudnn.benchmark = True
            
            threading.Thread(target=self.log_code).start()
            
            global ferr
            ferr = io.StringIO()
            with redirect_stderr(ferr):
                if tsk == "txt2img":
                    image = pipe(
                        POGPROMPT,
                        negative_prompt=NEGPROMPT,
                        generator=seed,
                        guidance_scale=SCALE,
                        num_inference_steps=STEPS,
                        width=WIDTH_SIZE,
                        height=HEIGHT_SIZE,
                        max_embeddings_multiples=10,
                        # callback=self.latents_callback, 
                        # callback_steps=4
                    ).images[0]
                        
    
                    file_path = OUTPUT_PATH + "[" + now + "][" + "[" + str(SEED) + "][" + combo_custom_set.get() + "][" + MODEL_NAME + "]" + ".png"
                
                elif tsk == "img2img":
                    # fpath = OUTPUT_PATH + lineup_files[img]
                    # lineup_files.sort(key = os.path.getmtime, reverse = True)
                    image = sub_lineup_data[img]
                    
                    rasio = 2.5
                    print("upscale and image to image:", image.width, "*", image.height, "->",int(image.width*rasio), "*",int(image.height*rasio))
                    self.dsply_log("END", "upscale and image to image:\n" + str(image.width) +" * " + str(image.height) + " -> " + str(image.width*rasio) + " * " + str(image.height*rasio))
                    
                    image = image.resize((int(image.width*rasio),int(image.height*rasio)))
                    
                    image = pipe(
                        prompt=POGPROMPT, 
                        negative_prompt = NEGPROMPT,
                        generator=seed,
                        image=image, 
                        strength=0.5, 
                        guidance_scale=SCALE,
                        num_inference_steps=STEPS,
                        max_embeddings_multiples=10,
                    ).images[0]
                 
                    
                    file_path = lineup_files[img].split(".")
                    file_path = file_path[0] + "i2i." + file_path[1]
            
            
            image.save(file_path)
            image.save("test.png")
            
            self.change_size(0)
  
            # playsound.playsound('pippi.mp3')
        
            SEED += seed_span[cs["seed"]["shift"]]
            
            pipe = ""
            
            
            # img = Image.open(file_path)
            # dicta = img._getexif()
            # print(dicta)
                                        
        button_generate["text"] = "▶︎"
        button_generate["fg"] = "#00AA00"
        print("Finished.")
        self.dsply_log("END", "finished.")
            
    
    def latents_callback(self, i, t, latents):       
        global pipe
        global tp
        global STEPS
        global now
        
        if tp <= i:
            tp = tp + STEPS/3
            vae = pipe.vae
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(1, 2, 0).numpy()
            image = pipe.numpy_to_pil(image)
            img = Image.new("RGB", size=(int(spin_width.get()),int(spin_height.get())))
            img.paste(image[0], box=(0,0))
            
            global OUTPUT_PATH
            global MODEL_NAME
            
            file_path = OUTPUT_PATH + "[" + now + "]["+ "[" + str(SEED) + "][" + combo_custom_set.get() + "][" + MODEL_NAME + "]" + ".png"
            img.save(file_path)
            img.save("test.png")

            self.change_size(0)
        
        
    def cp_progress(self, from_file, to_file):
        global cp_progress
        global prgrs_load
        global prgrs_var
        
        cp_progress = -1
        from_file_size = os.path.getsize(from_file)
        to_file_size = 0
        prev_progress = -1.0
        prgrs_var.set("0")
        
        while prgrs_var.get() < 100:
            if os.path.isfile(to_file):
                to_file_size = os.path.getsize(to_file)
            progress_rate = "{:.1f}".format(to_file_size / from_file_size * 100)

            if prev_progress != progress_rate:
                print("cp progress:" + progress_rate + "%")
                prgrs_var.set(progress_rate)
            prev_progress = progress_rate
            sleep(1)
    
    def dsply_log(self, place, in_text):
        global log_line
        global prg_generate
        global cp_progress
        global prgrs_var
        
        match place:
            case "END":
                try:
                    log_line.append(in_text + "\n")
                except:
                    log_line = []
            case "prg_generate":
                if prg_generate != -1 and prg_generate == len(log_line)-1:
                    log_line[prg_generate] = in_text + "\n"
                
                else:
                    prg_generate = len(log_line)
                    log_line.append(in_text + "\n")
            case "cp_progress":
                if cp_progress != -1 and cp_progress == len(log_line)-1:
                    log_line[cp_progress] = in_text + "\n"
                
                else:
                    cp_progress = len(log_line)
                    log_line.append(in_text + "\n")
            
        text_log.delete(1.0, tk.END)
        for line_str in log_line:
            text_log.insert(tk.END, line_str)
            text_log.see(tk.END)
    
    def win_activate(self, _):
        sub_view.lift()
        sub_lineup.lift()
    
            
    def test(self, ):
        print("test")
        
        
        
def main():
    global gui
    root = tk.Tk()
    gui = GUI(root)
    # root.attributes("-topmost", True)
    root.mainloop()
    

if __name__ == "__main__":
    main() 