OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.2426836) q[5];
rz(1.5554317) q[5];
rz(0.47629957) q[5];
rz(1.6935384) q[5];
rz(3.1764825) q[5];
rz(3.5864843) q[11];
rz(2.9604495) q[8];
rz(5.4981577) q[8];
rz(5.6636468) q[8];
rz(0.34964608) q[8];
cx q[5],q[8];
cx q[11],q[8];
rz(4.9487211) q[8];
rz(5.4474259) q[8];
cx q[8],q[5];
cx q[11],q[8];
rz(0.91524067) q[5];
rz(5.9322425) q[5];
rz(5.2177031) q[5];
rz(0.65415942) q[5];
rz(4.7230046) q[8];
rz(2.5159142) q[8];
