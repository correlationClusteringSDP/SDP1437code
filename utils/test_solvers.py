#!/usr/bin/env python3
"""
Test script to verify CVXPY and MOSEK installation.
Run this script to check if both solvers are working correctly.
"""

import sys

def test_imports():
    """Test if required packages can be imported."""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    
    success = True
    
    # Test numpy
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: FAILED - {e}")
        success = False
    
    # Test scipy
    try:
        import scipy
        print(f"✓ SciPy: {scipy.__version__}")
    except ImportError as e:
        print(f"✗ SciPy: FAILED - {e}")
        success = False
    
    # Test cvxpy
    try:
        import cvxpy as cp
        print(f"✓ CVXPY: {cp.__version__}")
    except ImportError as e:
        print(f"✗ CVXPY: FAILED - {e}")
        success = False
        return False
    
    # Test mosek
    try:
        import mosek
        # MOSEK doesn't always have __version__, try alternative methods
        try:
            version = mosek.__version__
        except AttributeError:
            try:
                # Try to get version from environment
                env = mosek.Env()
                version = "installed (version check unavailable)"
            except:
                version = "installed"
        print(f"✓ MOSEK: {version}")
    except ImportError as e:
        print(f"✗ MOSEK: FAILED - {e}")
        print("  Note: MOSEK is optional. SCS solver will be used if MOSEK is not available.")
    
    print()
    return success


def test_scs_solver():
    """Test SCS solver (open-source, no license needed)."""
    print("=" * 60)
    print("Testing SCS Solver")
    print("=" * 60)
    
    try:
        import cvxpy as cp
        import numpy as np
        
        # Simple test problem: minimize ||x||^2 subject to x >= 0, sum(x) = 1
        n = 5
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(x))
        constraints = [x >= 0, cp.sum(x) == 1]
        prob = cp.Problem(objective, constraints)
        
        # Solve with SCS
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-5)
        
        if prob.status == 'optimal':
            print(f"✓ SCS solver: SUCCESS")
            print(f"  Optimal value: {prob.value:.6f}")
            print(f"  Solution: {x.value}")
            print()
            return True
        else:
            print(f"✗ SCS solver: FAILED - Status: {prob.status}")
            print()
            return False
            
    except Exception as e:
        print(f"✗ SCS solver: FAILED - {e}")
        print()
        return False


def test_mosek_solver():
    """Test MOSEK solver (requires license)."""
    print("=" * 60)
    print("Testing MOSEK Solver")
    print("=" * 60)
    
    try:
        import cvxpy as cp
        import numpy as np
        import mosek
        
        # Check license
        try:
            env = mosek.Env()
            license_path = env.getlicensepath()
            print(f"  MOSEK license found at: {license_path}")
        except Exception as e:
            print(f"  Warning: Could not access MOSEK license: {e}")
            print("  MOSEK may still work if license is in default location.")
        
        # Simple test problem: minimize ||x||^2 subject to x >= 0, sum(x) = 1
        n = 5
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(x))
        constraints = [x >= 0, cp.sum(x) == 1]
        prob = cp.Problem(objective, constraints)
        
        # Solve with MOSEK
        prob.solve(solver=cp.MOSEK, verbose=False, eps=1e-5)
        
        if prob.status == 'optimal':
            print(f"✓ MOSEK solver: SUCCESS")
            print(f"  Optimal value: {prob.value:.6f}")
            print(f"  Solution: {x.value}")
            print()
            return True
        else:
            print(f"✗ MOSEK solver: FAILED - Status: {prob.status}")
            print()
            return False
            
    except ImportError:
        print("✗ MOSEK solver: NOT INSTALLED")
        print("  Install with: conda install -c mosek mosek")
        print()
        return False
    except Exception as e:
        print(f"✗ MOSEK solver: FAILED - {e}")
        print("  This might be due to:")
        print("    - Missing or invalid license file")
        print("    - License file not in correct location")
        print("    - License expired or invalid for this machine")
        print("  See SETUP_ENVIRONMENT.md for license setup instructions.")
        print()
        return False


def test_sdp_problem():
    """Test a simple SDP problem similar to the actual use case."""
    print("=" * 60)
    print("Testing SDP Problem (Similar to Actual Use Case)")
    print("=" * 60)
    
    try:
        import cvxpy as cp
        import numpy as np
        
        # Create a small SDP problem
        n = 3
        X = cp.Variable((n, n), symmetric=True)
        c = np.array([1.0, 2.0, 3.0])
        
        objective = cp.Maximize(cp.trace(X @ np.diag(c)))
        constraints = [
            X >> 0,  # Positive semidefinite
            cp.trace(X) == 1
        ]
        prob = cp.Problem(objective, constraints)
        
        # Try SCS first
        print("  Solving with SCS solver...")
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-5)
        
        if prob.status == 'optimal':
            print(f"  ✓ SCS: Optimal value = {prob.value:.6f}")
            scs_success = True
        else:
            print(f"  ✗ SCS: Failed - Status: {prob.status}")
            scs_success = False
        
        # Try MOSEK if available
        try:
            import mosek
            print("  Solving with MOSEK solver...")
            prob.solve(solver=cp.MOSEK, verbose=False, eps=1e-5)
            
            if prob.status == 'optimal':
                print(f"  ✓ MOSEK: Optimal value = {prob.value:.6f}")
                mosek_success = True
            else:
                print(f"  ✗ MOSEK: Failed - Status: {prob.status}")
                mosek_success = False
        except:
            print("  MOSEK: Not available or failed")
            mosek_success = False
        
        print()
        return scs_success or mosek_success
        
    except Exception as e:
        print(f"✗ SDP test: FAILED - {e}")
        print()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CVXPY and MOSEK Solver Test Suite")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Test imports
    results['imports'] = test_imports()
    if not results['imports']:
        print("ERROR: Required packages not installed. Please install CVXPY first.")
        sys.exit(1)
    
    # Test SCS solver
    results['scs'] = test_scs_solver()
    
    # Test MOSEK solver
    results['mosek'] = test_mosek_solver()
    
    # Test SDP problem
    results['sdp'] = test_sdp_problem()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Package Imports: {'✓ PASS' if results['imports'] else '✗ FAIL'}")
    print(f"SCS Solver:      {'✓ PASS' if results['scs'] else '✗ FAIL'}")
    print(f"MOSEK Solver:    {'✓ PASS' if results['mosek'] else '✗ FAIL (optional)'}")
    print(f"SDP Problem:     {'✓ PASS' if results['sdp'] else '✗ FAIL'}")
    print()
    
    # Final verdict
    if results['scs']:
        print("✓ SUCCESS: At least one solver (SCS) is working.")
        print("  You can run the SDP solver using SCS.")
        if results['mosek']:
            print("  MOSEK is also available and can be used for better performance.")
        else:
            print("  MOSEK is not available, but SCS will work fine.")
        return 0
    else:
        print("✗ FAILURE: No working solvers found.")
        print("  Please check your installation.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

