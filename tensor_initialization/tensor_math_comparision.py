import torch

#===================================================================================================#
#                   Tensor Math Comparision                                                         #
#===================================================================================================#

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

#Addition
z1 = torch.empty(3)
torch.add(x,y,out=z1)

z2 = torch.add(x,y)
z = x+y

#Subtraction
z = x - y
#division
z= torch.true_divide(x,y)

#inplace operations
t  = torch.zeros(3)
t.add_(x)
t+=x #t = t+x

#Exponentiation
z = x.pow(2)
z = x ** 2

#simple comparison
z = x>0
z = x<0
print("z =",z)

#matric multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2) #2x3
x3  =x1.mm(x2)

#matrix exponentiation
matrix_exp = torch.rand((5,5))
print("matrix_exp=",matrix_exp.matrix_power(3))

#dot product
z = torch.dot(x,y)
print(z)

#batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))

out_bmm = torch.bmm(tensor1,tensor2) #(batch,n,p)

#example of Broasdcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1-x2
z = x1 **x2

#Other useful tensor operators
sum_x = torch.sum(x,dim=0)
values,indices = torch.max(x,dim=0)
values,indices = torch.min(x,dim=0)

abs_x = torch.abs(x)
z = torch.argmax(x,dim=0)
z = torch.argmin(x,dim=0)
mean_x = torch.mean(x.float(),dim=0)
z = torch.eq(x,y)
print(mean_x)
print(z)














