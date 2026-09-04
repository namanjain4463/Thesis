import numpy as np; np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

print("="*70)
print("2A  DOES ROBOT-MODEL ERROR MASQUERADE AS CONTACT FORCE?  (momentum obs)")
print("="*70)
# true arm inertia, FREE motion (no contact, lambda=0):  tau = M qdd + h
n=3
B=rng.standard_normal((n,n)); M=B@B.T+2*np.eye(n)          # true mass matrix (SPD)
qdd=rng.standard_normal(n); h=rng.standard_normal(n)
tau = M@qdd + h                                            # actuator torque during FREE motion
# momentum observer with WRONG model (mass off by delta, bias exact for clarity)
for delta in [0.0, 0.02, 0.05, 0.10]:
    Mhat = M*(1+delta)
    r = Mhat@qdd + h - tau                                 # residual it will CALL contact force
    print("  model error %.0f%% -> phantom generalized force ||r|| = %.3f  (true contact = 0)"
          % (delta*100, np.linalg.norm(r)))
print("  => r = (Mhat-M) qdd = dM*qdd. Model error is INDISTINGUISHABLE from contact")
print("     force. Worse: during training this leaks into C_theta -> C becomes")
print("     robot-specific -> BREAKS the invariance/transfer claim. (sim: dM=0, clean.)")

print("\n"+"="*70)
print("2B  DOES sigma_min BLOW UP AT v=0 / SINGULAR CONTACT?  (compliance test)")
print("="*70)
# Delassus W = J M^-1 J^T. Make it NEAR-SINGULAR (more contact constraints than dof,
# or nearly parallel contact directions) -> rigid LCP is ill-posed there.
Minv=np.linalg.inv(M)
J=np.array([[1,0,0],[1,0,1e-3],[0,1,0]],float)             # rows 0,1 nearly parallel -> W near-singular
W=J@Minv@J.T
smin_rigid=np.linalg.svd(W,compute_uv=False).min()
print("  rigid contact:      sigma_min(W)          = %.3e   (near 0 -> ill-posed)"%smin_rigid)
for Rc in [1e2,1e3,1e4]:
    A=W+Rc*np.eye(W.shape[0])
    print("  compliant (R=%6.0f): sigma_min(W+R)        = %.3e   (bounded BELOW by R)"%(Rc,np.linalg.svd(A,compute_uv=False).min()))
print("  => the compliance/regularization we ALREADY use makes sigma_min(A) >= ~R,")
print("     so it does NOT blow up at v=0. The '2B trap' is defused BY the design.")
print("     Real residual issue is elsewhere: at contact MODE SWITCHES the active set")
print("     changes -> the GRADIENT d(lambda)/d(eta) jumps (a TRAINING-time issue),")
print("     and grazing/breaking contacts have genuinely small sigma_min -> the")
print("     certificate CORRECTLY reports low margin (that is it working, not failing).")
