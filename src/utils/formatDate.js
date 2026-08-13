const dateFormatter = new Intl.DateTimeFormat('en-GB', {
  day: '2-digit',
  month: 'short',
  year: 'numeric',
  timeZone: 'UTC',
});

export const formatDate = (date) => {
  const normalizedDate = typeof date === 'string' ? new Date(`${date} UTC`) : date;
  return dateFormatter.format(normalizedDate).toUpperCase();
};
