OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.2860452) q[5];
rz(2.4583473) q[5];
rz(3.4641986) q[5];
rz(5.4178612) q[5];
rz(1.035201) q[11];
rz(5.2878801) q[11];
rz(5.3628989) q[11];
rz(4.7529247) q[5];
cx q[8],q[11];
rz(0.1664508) q[11];
rz(2.454825) q[11];
rz(6.0495409) q[11];
rz(5.2625473) q[11];
cx q[5],q[8];
cx q[8],q[5];
cx q[11],q[8];
cx q[8],q[11];
rz(3.3119823) q[8];
cx q[5],q[8];
rz(0.19116485) q[8];
rz(5.7365978) q[8];
