const dollars = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
});

export function formatPrice(price: string | null): string {
  const value = price?.trim();
  if (!value) return 'Price unavailable';
  if (!/^\d+(?:\.\d+)?$/.test(value)) return value;

  const amount = Number(value);
  if (!Number.isFinite(amount)) return 'Price unavailable';
  return amount === 0 ? 'Free' : dollars.format(amount);
}
