import torch
from mamba_ssm import Mamba
from RevIN.RevIN import RevIN
import math
import copy
from einops.layers.torch import Rearrange
from skimage.transform import resize
import pywt
import numpy as np
import joblib

class Model(torch.nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.configs=configs
        self.configs_copy = copy.deepcopy(self.configs)
        if self.configs.task_name=="classification":
            self.patcher=ConversionLayer(
                d_in=self.configs.enc_in,
                l_in=self.configs.seq_len,
                l_out=self.configs.projected_space,
                d_out=self.configs.enc_in,
                im_size=self.configs.rescale_size,
                patch_size=self.configs.patch_size)
            if self.configs.no_rocket==1:
                self.projector=torch.nn.Linear(self.configs.seq_len,self.configs.projected_space)
            elif self.configs.half_rocket==1:
                self.projector=torch.nn.Linear(self.configs.seq_len,self.configs.projected_space//2)

            self.ln=torch.nn.LayerNorm(self.configs.projected_space*3)
            self.dropout=torch.nn.Dropout(self.configs.dropout)
            self.learnable_focus=torch.nn.Parameter(torch.tensor([self.configs.initial_focus]))
            self.gelu=torch.nn.GELU()
            self.mamba1 = torch.nn.ModuleList([
                Mamba(d_model=self.configs.projected_space*3,
                      d_state=self.configs.d_state,
                      d_conv=self.configs.dconv,
                      expand=self.configs.e_fact) for _ in range(self.configs.num_mambas)
                ])
            self.mamba2 = torch.nn.ModuleList([
                Mamba(d_model=self.configs.enc_in,
                      d_state=self.configs.d_state,
                      d_conv=self.configs.dconv,
                      expand=self.configs.e_fact) for _ in range(self.configs.num_mambas)
                ])

            self.flatten=torch.nn.Flatten(start_dim=1)
            self.classifier=torch.nn.Sequential(
                torch.nn.Linear(self.configs.projected_space*3,(self.configs.projected_space*3)//2),
                torch.nn.Dropout(self.configs.dropout),
                torch.nn.Linear((self.configs.projected_space*3)//2,self.configs.num_class)        
            )
            

    def classification(self,x_cwt,batch_x_features):    
        x_patched=self.patcher(x_cwt)
        if self.configs.no_rocket==1:
            x_projected=self.projector(batch_x_features)
        else:
            if self.configs.half_rocket==0:
                x_projected=batch_x_features
            else:
                x_projected=torch.cat([batch_x_features[:,:,:self.configs.projected_space//2],
                                       self.projector(batch_x_features[:,:,self.configs.projected_space:])
                                    ],dim=2)
        if self.configs.additive_fusion==1:
            x_fused=(self.learnable_focus*x_projected)+(2.0-self.learnable_focus)*x_patched
        else:
            x_fused=(self.learnable_focus*x_projected)*(2.0-self.learnable_focus)*x_patched
            
        
        x_fused=self.gelu(x_fused)
        concatenated_x=torch.cat([x_patched,x_fused,x_projected],dim=2)
        concatenated_x=self.ln(concatenated_x)

        if self.configs.num_mambas!=0:
            x1=concatenated_x.clone()

            for i in range(self.configs.num_mambas):
                x1=self.mamba1[i](x1)+x1.clone()

            if self.configs.only_forward_scan==0:
                concatenated_x=torch.flip(concatenated_x,dims=[self.configs.flip_dir])
                x1_flipped=concatenated_x.clone()
                for i in range(self.configs.num_mambas):
                    x1_flipped=self.mamba1[i](x1_flipped)+x1_flipped.clone()     
                if self.configs.reverse_flip==0:           
                    x1=x1+x1_flipped
                elif self.configs.reverse_flip==1:
                    x1=x1+torch.flip(x1_flipped,dims=[self.configs.flip_dir])
                concatenated_x=torch.flip(concatenated_x,dims=[self.configs.flip_dir])

            concatenated_x=torch.permute(concatenated_x,(0,2,1))
            x2=concatenated_x.clone()
            for i in range(self.configs.num_mambas):
                x2=self.mamba2[i](x2)+x2.clone()
            x2=torch.permute(x2,(0,2,1))
            
            if self.configs.only_forward_scan==0:
                x2_flipped=torch.flip(concatenated_x.clone(),dims=[self.configs.flip_dir])
                for i in range(self.configs.num_mambas):
                    x2_flipped=self.mamba2[i](x2_flipped)+x2_flipped.clone()     

                x2_flipped=torch.permute(x2_flipped,(0,2,1))
                if self.configs.reverse_flip==0:
                    x2=x2+x2_flipped 
                elif self.configs.reverse_flip==1:
                    x2=x2+torch.flip(x2_flipped,dims=[self.configs.flip_dir])
            x3=x2+x1
        else:
            x3=concatenated_x
        if self.configs.max_pooling==0:
            x3 = x3.mean(1)
        else:
            x3, _ = x3.max(1)
        x3=self.flatten(x3)
        x_logits=self.classifier(x3)
        return x_logits

    def forward(self,x_cwt,x_features):
         if self.configs.task_name=='classification':
             return self.classification(x_cwt,x_features)
       

class ConversionLayer(torch.nn.Module):
    def __init__(self,d_in,l_in,d_out,l_out,im_size,patch_size):
        super(ConversionLayer, self).__init__()
        self.d_in=d_in
        self.l_in=l_in
        self.d_out=d_out
        self.l_out=l_out
        self.im_size=im_size
        self.patch_size=patch_size

        self.patch_embedding=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.d_in, out_channels=self.d_out, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size), padding=0),
            Rearrange('b c h w -> b c (h w)')
        )
        self.projector=torch.nn.Linear((self.im_size//self.patch_size)*(self.im_size//self.patch_size),self.l_out)
        
    def forward(self,x):
        x=self.patch_embedding(x)
        x=self.projector(x)
        return x

         