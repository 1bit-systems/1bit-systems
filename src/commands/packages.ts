export async function installPackage(spec: string): Promise<void> {
  console.log(`  Install package — coming in Task 5 (spec: ${spec})`);
}
export async function removePackage(name: string): Promise<void> {
  console.log(`  Remove package — coming in Task 5 (name: ${name})`);
}
export async function listPackages(): Promise<void> {
  console.log("  List packages — coming in Task 5");
}
