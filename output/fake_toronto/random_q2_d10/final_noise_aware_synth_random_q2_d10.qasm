OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.4920237) q[8];
rz(2.8278083) q[11];
rz(5.0962599) q[8];
rz(2.9514925) q[8];
rz(1.190194) q[8];
rz(4.7630309) q[11];
cx q[11],q[8];
rz(4.5244779) q[8];
rz(4.170452) q[8];
rz(4.3087538) q[8];
rz(4.6162632) q[8];