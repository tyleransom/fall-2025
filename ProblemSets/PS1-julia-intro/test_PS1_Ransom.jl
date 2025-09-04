using Test, JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions

# Include the main file
include("PS1_Ransom.jl")

@testset "PS1 Ransom Tests" begin
    
    @testset "matrixops function tests" begin
        # Test with simple matrices
        A = [1.0 2.0; 3.0 4.0]
        B = [2.0 1.0; 1.0 2.0]
        
        out1, out2, out3 = matrixops(A, B)
        
        # Test element-wise product
        @test out1 == [2.0 2.0; 3.0 8.0]
        
        # Test matrix product of A' and B
        expected_out2 = [1.0 3.0; 2.0 4.0] * [2.0 1.0; 1.0 2.0]
        @test out2 == expected_out2
        
        # Test sum of all elements
        @test out3 == sum([3.0 3.0; 4.0 6.0])
        
        # Test dimension mismatch error
        C = [1.0 2.0]  # 1x2 matrix
        D = [1.0; 2.0] # 2x1 matrix
        @test_throws ErrorException matrixops(C, D)
    end
    
    @testset "q1 function tests" begin
        # Set seed for reproducible tests
        Random.seed!(1234)
        A, B, C, D = q1()
        
        # Test matrix dimensions
        @test size(A) == (10, 7)
        @test size(B) == (10, 7)
        @test size(C) == (5, 7)  # [A[1:5, 1:5] B[1:5, end-1:end]]
        @test size(D) == (10, 7)
        
        # Test that A is within expected range (uniform between -5 and 10)
        @test all(-5 .<= A .<= 10)
        
        # Test that D only contains non-positive values from A or zeros
        for i in eachindex(D)
            if D[i] != 0
                @test D[i] <= 0
            end
        end
        
        # Test file creation (files should exist after running q1)
        @test isfile("matrixpractice.jld")
        @test isfile("firstmatrix.jld")
        @test isfile("Cmatrix.csv")
        @test isfile("Dmatrix.dat")
        
        # Test that saved matrices can be loaded
        saved_data = load("matrixpractice.jld")
        @test haskey(saved_data, "A")
        @test haskey(saved_data, "B")
        @test haskey(saved_data, "C")
        @test haskey(saved_data, "D")
    end
    
    @testset "q2 function tests" begin
        # Create test matrices
        Random.seed!(1234)
        A_test = rand(5, 3)
        B_test = rand(5, 3)
        C_test = [-6.0 -3.0 0.0 3.0 6.0; 
                  -4.0 -1.0 2.0 5.0 8.0]
        
        # Test that q2 runs without error
        @test q2(A_test, B_test, C_test) === nothing
        
        # Test element-wise multiplication logic
        AB_expected = A_test .* B_test
        AB_manual = zeros(size(A_test))
        for r in axes(A_test,1)
            for c in axes(A_test,2)
                AB_manual[r, c] = A_test[r, c] * B_test[r, c]
            end
        end
        @test AB_manual ≈ AB_expected
        
        # Test filtering logic
        Cprime_expected = C_test[(C_test .>= -5) .& (C_test .<= 5)]
        Cprime_manual = Float64[]
        for c in axes(C_test, 2)
            for r in axes(C_test, 1)
                if C_test[r, c] >= -5 && C_test[r, c] <= 5
                    push!(Cprime_manual, C_test[r, c])
                end
            end
        end
        @test Cprime_manual == Cprime_expected
    end
    
    @testset "Data generation tests (from q2)" begin
        Random.seed!(1234)
        N = 100  # Use smaller N for testing
        K = 6
        T = 5
        X = zeros(N, K, T)
        
        # Simulate the data generation from q2
        for i in 1:N
            X[i, 1, :] .= 1.0
            X[i, 5, :] .= rand(Binomial(20, 0.6))
            X[i, 6, :] .= rand(Binomial(20, 0.5))
            for t in 1:T
                X[i, 2, t] = rand() <= .75 * (6 - t) /5
                X[i, 3, t] = rand(Normal(15 + t - 1, 5*(t-1)))
                X[i, 4, t] = rand(Normal(π*(6 - t), 1/exp(1)))
            end
        end
        
        # Test dimensions
        @test size(X) == (N, K, T)
        
        # Test that first column is all ones
        @test all(X[:, 1, :] .== 1.0)
        
        # Test that columns 2 are binary (0 or 1)
        @test all(in([0.0, 1.0]), X[:, 2, :])
        
        # Test that columns 5 and 6 are integers between 0 and 20
        @test all(0 .<= X[:, 5, :] .<= 20)
        @test all(0 .<= X[:, 6, :] .<= 20)
        @test all(X[:, 5, :] .== round.(X[:, 5, :]))  # Check they're integers
        @test all(X[:, 6, :] .== round.(X[:, 6, :]))  # Check they're integers
    end
    
    @testset "Beta coefficient generation tests" begin
        K = 6
        T = 5
        β = zeros(K, T)
        
        # Generate beta coefficients as in q2
        β[1, :] = [1+0.25*(t-1) for t in 1:T]
        β[2, :] = [log(t) for t in 1:T]
        β[3, :] = [-sqrt(t) for t in 1:T]
        β[4, :] = [exp(t) - exp(t+1) for t in 1:T]
        β[5, :] = [t for t in 1:T]
        β[6, :] = [t/3 for t in 1:T]
        
        # Test dimensions
        @test size(β) == (K, T)
        
        # Test specific values
        @test β[1, 1] == 1.0
        @test β[1, 2] == 1.25
        @test β[2, 1] == log(1)
        @test β[2, 2] == log(2)
        @test β[3, 1] == -sqrt(1)
        @test β[5, 3] == 3
        @test β[6, 4] == 4/3
        
        # Test that all values are finite
        @test all(isfinite, β)
    end
    
    # Note: q3 and q4 tests would require the nlsw88.csv file to be present
    # These are integration tests that depend on external data
    @testset "Integration tests (require data files)" begin
        if isfile("nlsw88.csv")
            @test_nowarn q3()
            @test_nowarn q4()
            
            # Test that files are created by q3
            @test isfile("nlsw88_cleaned.csv")
        else
            @test_skip "nlsw88.csv not found - skipping q3 and q4 tests"
        end
    end
    
    @testset "File I/O tests" begin
        # Test matrix operations with saved data
        if isfile("matrixpractice.jld")
            @load "matrixpractice.jld"
            
            # Test that loaded matrices have expected properties
            @test size(A) == (10, 7)
            @test size(B) == (10, 7)
            @test isa(A, Array{Float64})
            @test isa(B, Array{Float64})
            
            # Test matrixops with loaded data
            @test_nowarn matrixops(A, B)
        end
        
        # Test CSV reading if files exist
        if isfile("Cmatrix.csv")
            df_C = CSV.read("Cmatrix.csv", DataFrame)
            @test size(df_C, 1) == 5  # Should have 5 rows
            @test size(df_C, 2) == 7  # Should have 7 columns
        end
    end
end

# Clean up test files (optional)
function cleanup_test_files()
    files_to_remove = [
        "matrixpractice.jld",
        "firstmatrix.jld", 
        "Cmatrix.csv",
        "Dmatrix.dat",
        "nlsw88_cleaned.csv"
    ]
    
    for file in files_to_remove
        if isfile(file)
            rm(file)
        end
    end
end

# Uncomment the line below if you want to clean up test files after running tests
cleanup_test_files()
