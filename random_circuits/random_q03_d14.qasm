OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rz(0.34924142) q[0];
cx q[1],q[2];
cx q[0],q[1];
rz(2.8724301) q[2];
cx q[2],q[0];
rz(4.2594015) q[1];
cx q[0],q[1];
rz(2.3498817) q[2];
rz(1.8142206) q[0];
cx q[1],q[2];
cx q[0],q[1];
rz(1.8925316) q[2];
cx q[0],q[1];
rz(3.9250534) q[2];
rz(4.4607344) q[1];
cx q[2],q[0];
rz(0.029739236) q[0];
rz(1.7863257) q[1];
rz(0.0097703425) q[2];
cx q[2],q[1];
rz(4.0200174) q[0];
rz(5.0071344) q[1];
rz(3.0474655) q[0];
rz(3.6093031) q[2];
cx q[1],q[0];
rz(2.3737352) q[2];
rz(5.5592642) q[1];
cx q[2],q[0];
rz(3.2242048) q[0];
rz(2.0828377) q[2];
rz(0.83365313) q[1];
