OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.6867621) q[0];
rz(0.02427648) q[0];
rz(3.6930929) q[1];
rz(1.7619925) q[0];
rz(0.15904019) q[1];
cx q[0],q[1];
rz(3.7562905) q[1];
rz(0.45669669) q[1];
rz(4.1878102) q[1];
rz(1.8904751) q[1];
rz(5.6514808) q[1];
cx q[1],q[0];