import { describe, expect, it } from 'vitest';
import { formatPrice } from './formatPrice';

describe('formatPrice', () => {
  it.each([
    ['4.99', '$4.99'],
    ['10', '$10.00'],
    ['1234.5', '$1,234.50'],
    [' 1.99 ', '$1.99'],
    ['0.0', 'Free'],
    [null, 'Price unavailable'],
    [' ', 'Price unavailable'],
    ['$4.99', '$4.99'],
    ['₹480', '₹480'],
    ['Free', 'Free'],
  ])('formats %s as %s', (price, expected) => {
    expect(formatPrice(price)).toBe(expected);
  });
});
