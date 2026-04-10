export function formatInteger(value: number): string {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value)
}

export function formatDecimal(value: number, fractionDigits = 4): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(value)
}

export function formatOptionalNumber(
  value: number | null | undefined,
  fractionDigits = 4,
): string {
  return value == null ? '-' : formatDecimal(value, fractionDigits)
}

export function formatPercent(value: number, fractionDigits = 1): string {
  return `${new Intl.NumberFormat('en-US', {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(value * 100)}%`
}
