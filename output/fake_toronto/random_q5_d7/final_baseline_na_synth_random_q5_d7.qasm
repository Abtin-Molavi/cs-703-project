OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.25918975) q[23];
rz(3.3436504) q[24];
rz(5.2714007) q[24];
rz(4.6891413) q[24];
rz(1.3134001) q[24];
rz(4.1657225) q[25];
rz(5.2873486) q[25];
rz(5.2798083) q[26];
rz(4.4785084) q[26];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[24],q[23];
rz(0.90136902) q[23];
cx q[23],q[21];
rz(3.7128807) q[21];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[24],q[23];
rz(1.3053253) q[23];
cx q[25],q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[23],q[24];
cx q[26],q[25];
rz(2.3533652) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
cx q[21],q[23];
rz(3.6280043) q[24];
cx q[25],q[26];
rz(6.0157284) q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[25],q[24];
