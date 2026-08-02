export default function Badge({ label, value, color }: { label: string; value: string; color: "blue" | "green" | "amber" | "gray" }) {
    const valueColor = {
      blue: "bg-blue-500",
      green: "bg-emerald-500",
      amber: "bg-amber-500",
      gray: "bg-neutral-600",
    }[color];
  
    return (
      <span className="inline-flex text-xs font-mono rounded overflow-hidden">
        <span className="bg-neutral-800 text-neutral-300 px-2 py-1">{label}</span>
        <span className={`${valueColor} text-black font-medium px-2 py-1`}>{value}</span>
      </span>
    );
  }