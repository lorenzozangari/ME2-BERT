
import torch
import torch.nn.functional as F

class FFClassifier(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, n_classes, dropout=0.3):
        super(FFClassifier, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(True),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim, n_classes))

    def forward(self, input):

        return self.model(input)


class Encoder(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(hidden_dim, latent_dim, bias=True)
        self.prelu = torch.nn.PReLU()

    def forward(self, x):
        x = self.prelu(self.fc1(x))
        x = self.fc2(x)
        return x
    


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(latent_dim, hidden_dim, bias=True)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim, bias=True)
        self.prelu = torch.nn.PReLU()



    def forward(self, x):
        x = self.prelu(self.fc1(x))
        return self.fc2(x)
 


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.layer_norm = torch.nn.LayerNorm(latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.dropout = torch.nn.Dropout(0.3)


    def forward(self, x):
        encoded = self.encoder(self.dropout(x))
        encoded = self.layer_norm(encoded)
        decoded = self.decoder(encoded)
        decoded = decoded
        return encoded, decoded
 
 
  

class AutoEncoderLoss(torch.nn.Module):
    
    def __init__(self, device, sparsity_penalty=1e-2, sparsity_target=0.05, rec_loss=True, sparse_loss=True, cosine_loss=False, lambda_cosine=1e-2):
        super(AutoEncoderLoss, self).__init__()
        self.sparsity_penalty = sparsity_penalty
        self.sparsity_target = sparsity_target
        self.rec_loss = torch.nn.MSELoss() if rec_loss else None
        self.cosine_loss = torch.nn.CosineEmbeddingLoss() if cosine_loss else None
        self.sparse_loss =sparse_loss
        self.device = device
        self.lambda_cosine = lambda_cosine
        
    def forward(self, output, target, latent):
        loss = 0
        if self.rec_loss:
            loss += self.rec_loss(output, target)
        if self.cosine_loss:
            loss += self.lambda_cosine * self.cosine_loss(output, target, torch.ones(output.size(0)).to(self.device))
        if self.sparse_loss:
            rho_hat = torch.mean(torch.sigmoid(latent), dim=0)
            sparsity_loss = torch.sum(self.sparsity_target * torch.log(self.sparsity_target / rho_hat) + 
                                  (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - rho_hat)))
            loss = loss + self.sparsity_penalty*sparsity_loss
            
        return loss
        

class GatedCombination(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(GatedCombination, self).__init__()
        self.embedding_dim = embedding_dim

        self.forget_gate = torch.nn.Linear(embedding_dim, embedding_dim)
        self.input_gate = torch.nn.Linear(embedding_dim, embedding_dim)
        self.output_gate = torch.nn.Linear(embedding_dim, embedding_dim)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, frozen_output, finetuned_output):
        forget_gate = self.sigmoid(self.forget_gate(frozen_output))
        input_gate = self.sigmoid(self.input_gate(finetuned_output))
        
        combined = forget_gate * frozen_output + input_gate * finetuned_output
        
        output_gate = self.sigmoid(self.output_gate(combined))
        
        gated_output = output_gate * self.tanh(combined)
        
        return gated_output
    