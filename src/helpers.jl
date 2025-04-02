# per qubit drive in drive basis
function a_q(n_levels)
    a = zeros((n_levels,n_levels))
    for i ∈ 1:n_levels-1
        a[i,i+1] = √i            
    end
    return a
end
function a_fullspace(n_sites,n_levels,eig_basis)
    a_k = a_q(n_levels) 
    a=Vector{Matrix{Float64}}(undef, n_sites)
    for k in 1:n_sites 
        A = ones(1,1)                   
        I = one(a_k)                
        for i ∈ 1:n_sites
             # `a_k` acts on sites `k`, `I` acts on other sites
            if i==k
                A= kron(A, a_k)
            else
                A= kron(A, I)
            end   
        end
        a[k]=A
    end 
    if eig_basis == true
        for k in 1:n_sites
            a[k] = eigbasis'*a[k]*eigbasis
        end
        
    end                          
    return a
end